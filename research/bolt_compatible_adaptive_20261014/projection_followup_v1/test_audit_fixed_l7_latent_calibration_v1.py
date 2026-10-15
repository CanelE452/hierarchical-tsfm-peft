"""Independent synthetic CPU audit fixtures; no real result, model, F, GPU or producer execution."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import hashlib
import json
import tempfile
import unittest

import numpy as np

import audit_fixed_l7_latent_calibration_v1 as audit

audit.initialize_numpy()


def npz(path, arrays):
    with path.open("wb") as handle: np.savez(handle, **arrays)
    return {**audit.base.file_record(path), "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()} for k, v in arrays.items()}}


def put(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8"); return audit.base.file_record(path)


def fixture(k=2, c=3, seed=72):
    rng = np.random.default_rng(seed); past = rng.normal(size=(4, 40, k)).astype(np.float32); future = rng.normal(size=(4, 48, k)).astype(np.float32); d = rng.normal(scale=.1, size=(7*k, c)).astype(np.float32); current = rng.normal(size=(4, c)).astype(np.float32)
    baseline = audit.direct_decode(past, future, current, d); mask = np.ones_like(baseline, bool); mask[:3, :, 1] = False
    raw = {"raw_last": current, "target": baseline.copy(), "observed": mask, "std": np.arange(1, c+1, dtype=np.float32), "origins": np.arange(4, dtype=np.int64)}
    raw["target"][~mask] = np.nan
    return past, future, d, raw, {"latent_history": past, "latent_future": future, "prediction": baseline}


def fit_metadata(calculated):
    return {"FIT_channel_observed_counts": calculated["counts"].tolist(), "active_FIT_channels": int((calculated["counts"] > 0).sum()), "observed_FIT_rows": int(calculated["counts"].sum()), "row_weight_sum": calculated["row_weight_sum"], "exact_zero_FIT_columns": np.flatnonzero(~calculated["active"]).tolist(), "zero_lambda_numeric_rank": calculated["rank"], "SVD_fits": int(calculated["active"].any()), "coefficient_candidates": 7, "FIT_column_RMS_only": True, "response_is_actual_cached_baseline_residual": True, "identity_centered_regularization_on_FIT_RMS_scaled_coefficients": True, "column_or_response_centered": False, "positive_lambda_discards_small_singular_values": False, "coefficients_identified_unique_latent_causal_effects_claimed": False, "actual_FP32_equals_linear_surrogate_required": False, "zero_lambda_relative_cutoff": 1e-12, "lambdas": list(audit.LAMBDAS), "E_D_F_fits": 0, "FP64_design_check_atol": 1e-10, "FP64_design_check_rtol": 5e-12, "independent_FP32_deployment_parity_atol": 1e-5, "independent_FP32_deployment_parity_rtol": 2e-6}


def coefficients(beta, k, family):
    with np.errstate(over="ignore"): beta32 = beta.astype(np.float32)
    result = {"coefficients_FP64": beta.copy(), "a_FP64": beta[:k].copy(), "b_FP64": beta[k:].copy() if family == "SCALE_BIAS" else np.zeros(k, np.float64)}
    if np.isfinite(beta32).all(): result.update(coefficients_FP32=beta32, a_FP32=beta32[:k].copy(), b_FP32=beta32[k:].copy() if family == "SCALE_BIAS" else np.zeros(k, np.float32))
    return result


def candidate(coeff, family, lam, calculated, *, identity=False):
    beta = coeff["coefficients_FP64"]; theta = beta[calculated["active"]]*calculated["scales"][calculated["active"]]; residual = calculated["response"]-calculated["matrix"]@theta
    meta = {"identity_baseline_byte_exact_separate_path": True} if identity else {"standardized_coefficient_L2": float(np.linalg.norm(theta)), "coefficient_L2": float(np.linalg.norm(beta)), "coefficient_max_abs": float(np.abs(beta).max()), "surrogate_FIT_macro_MSE": float(residual@residual), "ridge_penalty": float(lam*(theta@theta)), "nonfinite_scalar_fields": [], "coefficient_clipping_applied": False, "negative_total_scale_channels": np.flatnonzero(1+coeff["a_FP64"] < 0).tolist(), "singular_filter": "relative_cutoff_1e-12" if lam == 0 else "all_singular_values_sigma_over_sigma2_plus_lambda"}
    return {"id": "IDENTITY" if identity else "RIDGE_"+format(lam, "g"), "kind": "IDENTITY" if identity else "RIDGE", "family": family, "lambda": None if identity else lam, "status": "COMPLETE" if "a_FP32" in coeff else "FAILED_NONFINITE_COEFFICIENTS_PRESERVED", "metadata": meta, "coefficient_value_sha256": audit.base.state_hash(coeff)}


def evaluation(coeff, cand, raw, cached, decoder, family, binding=None):
    identity = cand["kind"] == "IDENTITY"; u = cached["latent_future"] if identity else cached["latent_future"]+coeff["a_FP32"]*(cached["latent_future"]-cached["latent_history"][:, -1:])+coeff["b_FP32"]
    prediction = cached["prediction"].copy() if identity else audit.direct_decode(cached["latent_history"], u, raw["raw_last"], decoder)
    phi = audit.direct_design(cached["latent_history"], cached["latent_future"], decoder, family); surrogate = cached["prediction"].astype(np.float64)+phi@coeff["coefficients_FP64"]; error = prediction.astype(np.float64)-surrogate
    arrays = {"prediction_FP32": prediction, "latent_future_calibrated_FP32": u.copy(), "surrogate_prediction_FP64": surrogate, "actual_minus_surrogate_FP64": error, "independent_NumPy_future_FP32": u.copy(), "independent_NumPy_prediction_FP32": audit.direct_decode(cached["latent_history"], u, raw["raw_last"], decoder), "origins": raw["origins"]}
    complete = np.broadcast_to(raw["observed"].all(2, keepdims=True), raw["observed"].shape)
    entry = {"status": "COMPLETE", "input_source": binding or {}, "cached_Bolt_source": binding or {}, "coefficient_value_sha256": audit.base.state_hash(coeff), "independent_FP32_deployment_parity": {k: {"pass": True, "atol": 1e-5, "rtol": 2e-6, "max_abs": 0., "rmse": 0.} for k in ("latent_future", "C_prediction")}, "identity_checks": {"identity_byte_exact": identity, "batch_size": 32, "identity_object_checks": 1 if identity else 0}, "surrogate_diagnostics": {"max_abs_actual_FP32_minus_linear_surrogate": float(np.abs(error).max()), "RMS_actual_FP32_minus_linear_surrogate": float(np.sqrt(np.mean(error*error))), "per_channel_max_abs": np.max(np.abs(error), axis=(0, 1)).tolist(), "surrogate_zero_error_required": False, "linear_surrogate_is_actual_deployment_claimed": False}, "FP64_surrogate_is_not_deployment_parity_target": True, "target_future_is_loss_or_evaluation_only": True, "scores_all_observed": audit.prior.scores(prediction, raw["target"], raw["observed"], raw["std"]), "scores_complete_mask": audit.prior.scores(prediction, raw["target"], complete, raw["std"]) if complete.any() else {"status": "NO_COMPLETE_C"}, "surrogate_scores_all_observed": audit.prior.scores(surrogate, raw["target"], raw["observed"], raw["std"])}
    return entry, arrays


def full_fixture(folder):
    rng = np.random.default_rng(912); c, k = 17, 4; frozen = {"center": np.zeros(c, np.float32), "E": rng.normal(scale=.2, size=(c, k)).astype(np.float32), "D": rng.normal(scale=.1, size=(7*k, c)).astype(np.float32)}
    state_binding = npz(folder/"state.npz", frozen); raws, caches, raw_bindings, cache_bindings = {}, {}, {}, {}
    for split, n in (("fit", 4), ("tune", 3), ("val", 2)):
        history = rng.normal(size=(n, 512, c)).astype(np.float32); past = history@frozen["E"]; future = rng.normal(size=(n, 48, k)).astype(np.float32); current = history[:, -1].copy(); baseline = audit.direct_decode(past, future, current, frozen["D"]); mask = np.ones_like(baseline, bool); mask[0, :20, 1] = False
        raw = raws[split] = {"history": history, "raw_last": current, "target": baseline.copy(), "observed": mask, "std": np.ones(c, np.float32), "mean": np.zeros(c, np.float32), "origins": np.arange(n, dtype=np.int64)}; raw["target"][~mask] = np.nan
        caches[split] = {**raw, "latent_history": past, "latent_future": future, "prediction": baseline}
        raw_bindings[split] = npz(folder/(split+"_raw.npz"), raw); cache_bindings[split] = npz(folder/(split+"_cache.npz"), caches[split])
    science = {"status": "COMPLETE", "protected_TEST_accessed": False, "datasets": {"robin": {"spaces": {"PCA": {"conditions": {"L7_RECON": {"splits": {s: {"input_source": raw_bindings[s]} for s in raws}}}}}}}}
    science_binding = put(folder/"science.json", science)
    native = {"status": "COMPLETE", "protected_TEST_accessed": False, "F_parent_signatures_reused": {"before": "same", "after": "same"}, "causal_lag_diagnostic": science_binding, "states": {s: {"frozen_state_source": state_binding, "Bolt": {split: {"artifact": cache_bindings[split]} for split in raws}} for s in audit.STATES}}
    parents = {"matched_l7_native_diagnostic": put(folder/"native.json", native)}
    parents["matched_l7_native_audit"] = put(folder/"native_audit.json", {"status": "PASS", "protected_TEST_accessed": False, "diagnostic": parents["matched_l7_native_diagnostic"]})
    parents["matched_l7_native_cost"] = put(folder/"cost.json", {"status": "COMPLETE", "protected_TEST_accessed": False})
    parents["matched_l7_native_cost_audit"] = put(folder/"cost_audit.json", {"status": "PASS", "protected_TEST_accessed": False, "cost": parents["matched_l7_native_cost"]})
    report = folder/"REPORT.md"; report.write_text("synthetic accepted report", encoding="utf-8"); parents["parent_report"] = audit.base.file_record(report)
    parents["parent_report_manifest"] = put(folder/"MANIFEST.json", {"status": "COMPLETE", "protected_TEST_accessed": False})
    parents["parent_report_audit"] = put(folder/"report_audit.json", {"status": "PASS", "protected_TEST_accessed": False, "report": parents["parent_report"], "manifest": parents["parent_report_manifest"], "inputs": {key: parents[source] for key, source in (("train", "matched_l7_native_diagnostic"), ("audit", "matched_l7_native_audit"), ("cost", "matched_l7_native_cost"), ("cost_audit", "matched_l7_native_cost_audit"))}})
    controller = audit.base.file_record(audit.HERE/"run_fixed_l7_latent_calibration_v1.py"); module = audit.base.file_record(audit.HERE/"fixed_l7_latent_calibration_v1.py"); plan = folder/"PLAN.md"; plan.write_text("synthetic plan", encoding="utf-8")
    result = {"status": "COMPLETE", **parents, "causal_lag_diagnostic": science_binding, "controller_source": controller, "module_source": module, "plan": audit.base.file_record(plan), "source_records": [controller, module], "input_records": [*raw_bindings.values(), *cache_bindings.values(), state_binding], "protected_prior_records": list(parents.values()), "new_F_forward_calls": 0, "new_F_model_objects_loaded": 0, "new_optimizer_updates": 0, "new_E_D_training_updates": 0, "native_head_weights_loaded": False, "new_backbone_training": False, "new_F_state_hash_computed": False, "whole_deployment_cost_measured": False, "independent_holdout": False, "strict_K_only": False, "protected_TEST_accessed": False, "F_parent_signatures_reused": native["F_parent_signatures_reused"], "cached_Bolt_contexts_reused": 9, "cached_baseline_redecode_checks": 9, "selected_FIT_TUNE_predictions_reused_without_new_inference": True, "resource_terminal": {"cpu_seconds": 1., "wall_seconds": 2.}, "states": {}, "resolved_selection_locks": 6, "all_6_locks_before_DEV": True, "failed_paths": [], "failed_candidate_paths": [], "fit_conditions_completed": 6, "SVD_fits_completed": 6, "ridge_candidates_generated": 42, "identity_comparisons_generated": 6, "candidate_FIT_TUNE_evaluations_completed": 96, "selected_DEV_evaluations_completed": 6}
    authorization = put(folder/"scope.json", {"status": "APPROVED_HUMAN_UNCAPPED_CONTINUATION", "latest_human_override": "상한내수행하지 말고 쭉해줘"})
    continuation = put(folder/"continuation.json", {"authorization_sha256": authorization["sha256"], "runtime_sha256": audit.prior.file_sha(audit.HERE/"continuation_runtime_v1.py")})
    receipt = {"status": "COMPLETE", "gpu_process": True, "process_cpu_seconds": 1., "wall_seconds": 2., "authorization": authorization, "continuation_manifest": continuation, "original_budget_records_unchanged": True, "elapsed_or_cumulative_budget_caps": None}
    result["resource_terminal"] = {**put(folder/"receipt.json", receipt), **receipt}
    locks = []
    for state in audit.STATES:
        owner = result["states"][state] = {"frozen_state_source": state_binding, "E_previously_supervised_by_FIT_future": state != "RECON_CP0", "D_previously_supervised_by_FIT_future": False, "families": {}}
        for family in audit.FAMILIES:
            calc = audit.independent_fit(caches["fit"]["latent_history"], caches["fit"]["latent_future"], frozen["D"], caches["fit"]["prediction"], raws["fit"]["target"], raws["fit"]["observed"], family); label = state+family
            row = owner["families"][family] = {"status": "COMPLETE", "state": state, "family": family, "fit_metadata": fit_metadata(calc), "fit_artifact": npz(folder/(label+"_fit.npz"), {"column_scales_FP64": calc["scales"], "active_columns": calc["active"], "singularvalues_FP64": calc["singular"]}), "FIT_input_source": raw_bindings["fit"], "FIT_cached_Bolt_source": cache_bindings["fit"], "frozen_state_source": state_binding, "candidates": []}
            for index in range(8):
                beta = np.zeros(k if family == "SCALE_ONLY" else 2*k) if index == 0 else calc["solutions"][index-1]["coefficients"]; coeff = coefficients(beta, k, family); item = candidate(coeff, family, None if index == 0 else audit.LAMBDAS[index-1], calc, identity=index == 0)
                item["coefficient_artifact"] = npz(folder/(label+item["id"]+"_coef.npz"), coeff)
                for split in ("fit", "tune"):
                    entry, arrays = evaluation(coeff, item, raws[split], caches[split], frozen["D"], family); entry.update(input_source=raw_bindings[split], cached_Bolt_source=cache_bindings[split], artifact=npz(folder/(label+item["id"]+split+".npz"), arrays)); item[split.upper()] = entry
                item["TUNE_MSE"] = item["TUNE"]["scores_all_observed"]["normalized"]["mse"]; row["candidates"].append(item)
            winner = audit.selected(row["candidates"]); row.update(selected_candidate=winner["id"], selected_kind=winner["kind"], selected_lambda=winner["lambda"], selected_TUNE_MSE=winner["TUNE_MSE"], selected_coefficients=winner["coefficient_artifact"], selected_coefficient_value_sha256=winner["coefficient_value_sha256"], FIT=winner["FIT"], TUNE=winner["TUNE"])
            lock = put(folder/(label+"_lock.json"), {**row, "DEV_opened_before_lock": False, "sealed_unix": 0.}); row["selection_lock"] = lock; locks.append({"state": state, "family": family, "status": "COMPLETE", "selection_lock": lock})
            entry, arrays = evaluation(coefficients(np.zeros(k if family == "SCALE_ONLY" else 2*k), k, family), winner, raws["val"], caches["val"], frozen["D"], family); entry.update(input_source=raw_bindings["val"], cached_Bolt_source=cache_bindings["val"], artifact=npz(folder/(label+"_DEV.npz"), arrays)); row["DEV"] = entry
    result["selection_registry"] = put(folder/"registry.json", {"status": "ALL_SIX_RESOLVED_BEFORE_DEV", "DEV_opened_before_seal": False, "sealed_unix": 0., "conditions": locks})
    binding = put(folder/"RESULT.json", result)
    return binding, result, {"RESULT_SHA": binding["sha256"], "CONTROLLER_SHA": controller["sha256"], "PARENT_SHA": {k: parents[k]["sha256"] for k in audit.PARENT_SHA}}


class FixedL7CalibrationAuditTests(unittest.TestCase):
    def test_closed_actual_gate_before_reads_and_no_producer_import(self):
        with patch.object(audit, "RESULT_SHA", None), patch.object(audit.base, "file_record", side_effect=AssertionError("closed file")), self.assertRaisesRegex(Exception, "NOT_YET_REGISTERED"): audit.metadata("missing", lambda: None)
        text = Path(audit.__file__).read_text(encoding="utf-8"); self.assertNotIn("import fixed_l7_latent_calibration_v1", text); self.assertNotIn("import run_fixed_l7_latent_calibration_v1", text)

    def test_future_only_direct_design_and_lag32_plateau(self):
        p, u, d, raw, cache = fixture(); a = np.array([.3, -.6]); b = np.array([.1, -.2]); current = raw["raw_last"].astype(np.float64)
        def real(future):
            out = np.broadcast_to(current[:, None], raw["target"].shape).copy(); k = future.shape[-1]
            for h in range(48):
                for li, lag in enumerate(audit.LAGS): out[:, h] += ((future[:, h-lag] if h >= lag else p[:, p.shape[1]+h-lag]).astype(np.float64)-p[:, -1-lag].astype(np.float64))@d[li*k:(li+1)*k].astype(np.float64)
            return out
        changed = u.astype(np.float64)+a*(u.astype(np.float64)-p[:, -1:].astype(np.float64))+b
        np.testing.assert_allclose(audit.direct_design(p, u, d, "SCALE_BIAS")@np.r_[a, b], real(changed)-real(u), atol=1e-10, rtol=5e-12)
        d[:] = 0; d[-2:] = 1; phi = audit.direct_design(p, u, d, "SCALE_BIAS"); np.testing.assert_array_equal(phi[:, :32], 0)

    def test_augmented_solver_known_solution_mask_independence_and_zero_columns(self):
        p, u, d, raw, cache = fixture(); phi = audit.direct_design(p, u, d, "SCALE_BIAS"); truth = np.array([.2, -.4, .1, -.2]); target = (cache["prediction"].astype(np.float64)+phi@truth).astype(np.float32); target[~raw["observed"]] = np.nan
        calc = audit.independent_fit(p, u, d, cache["prediction"], target, raw["observed"], "SCALE_BIAS"); np.testing.assert_allclose(calc["solutions"][0]["coefficients"], truth, atol=1e-7, rtol=1e-7); self.assertAlmostEqual(calc["row_weight_sum"], 1.)
        target[~raw["observed"]] = 1e30; again = audit.independent_fit(p, u, d, cache["prediction"], target, raw["observed"], "SCALE_BIAS"); np.testing.assert_array_equal(calc["scales"], again["scales"])
        for x, y in zip(calc["solutions"], again["solutions"]): np.testing.assert_array_equal(x["coefficients"], y["coefficients"])
        d[1::2] = 0; zero = audit.independent_fit(p, u, d, cache["prediction"], target, raw["observed"], "SCALE_ONLY"); self.assertFalse(zero["active"][1]); self.assertTrue(all(v["coefficients"][1] == 0 for v in zero["solutions"]))
        d[:] = 0; empty = audit.independent_fit(p, u, d, cache["prediction"], target, raw["observed"], "SCALE_ONLY"); self.assertEqual(empty["rank"], 0); self.assertEqual(len(empty["singular"]), 0)

    def test_coefficient_cast_hash_scalar_and_failed_FP32_metadata(self):
        p, u, d, raw, cache = fixture(); family = "SCALE_BIAS"; calc = audit.independent_fit(p, u, d, cache["prediction"], raw["target"], raw["observed"], family); coeff = coefficients(calc["solutions"][0]["coefficients"], 2, family); item = candidate(coeff, family, 0., calc)
        self.assertTrue(audit.coefficient_check(item, coeff, calc["solutions"][0]["coefficients"], 2, family)); audit.scalar_diagnostics(item, coeff, calc)
        bad = copy.deepcopy(coeff); bad["a_FP32"][0] = .1
        with self.assertRaisesRegex(Exception, "a32_partition"): audit.coefficient_check(item, bad, calc["solutions"][0]["coefficients"], 2, family)
        huge = coefficients(np.array([1e40, 0.]), 2, "SCALE_ONLY"); failed = {"kind": "RIDGE", "family": "SCALE_ONLY", "status": "FAILED_NONFINITE_COEFFICIENTS_PRESERVED", "coefficient_value_sha256": audit.base.state_hash(huge)}
        self.assertFalse(audit.coefficient_check(failed, huge, np.array([1e40, 0.]), 2, "SCALE_ONLY"))

    def test_actual_FP32_own_split_scores_and_nonzero_surrogate_error(self):
        p, u, d, raw, cache = fixture(); family = "SCALE_BIAS"; calc = audit.independent_fit(p, u, d, cache["prediction"], raw["target"], raw["observed"], family); coeff = coefficients(np.array([.3, -.8, .1, -.2]), 2, family); item = candidate(coeff, family, 1., calc)
        entry, arrays = evaluation(coeff, item, raw, cache, d, family); self.assertGreater(audit.prediction_check(entry, arrays, item, coeff, cache, raw, d, family), 0); self.assertGreater(entry["surrogate_diagnostics"]["max_abs_actual_FP32_minus_linear_surrogate"], 0)
        for mutation in (lambda e, a: e["scores_all_observed"]["normalized"]["observed_counts"].__setitem__(1, 192), lambda e, a: a["prediction_FP32"].__setitem__((0, 0, 0), 99.), lambda e, a: e["surrogate_diagnostics"].update(surrogate_zero_error_required=True)):
            bad_entry, bad_arrays = copy.deepcopy(entry), copy.deepcopy(arrays); mutation(bad_entry, bad_arrays)
            with self.assertRaises(Exception): audit.prediction_check(bad_entry, bad_arrays, item, coeff, cache, raw, d, family)

    def test_identity_then_larger_lambda_and_source_six_lock_contract(self):
        rows = [{"kind": "RIDGE", "lambda": v, "status": "COMPLETE", "TUNE_MSE": 1.} for v in audit.LAMBDAS]; self.assertEqual(audit.selected(rows)["lambda"], 10.)
        rows.append({"kind": "IDENTITY", "lambda": None, "status": "COMPLETE", "TUNE_MSE": 1.}); self.assertEqual(audit.selected(rows)["kind"], "IDENTITY")
        text = (audit.HERE/"run_fixed_l7_latent_calibration_v1.py").read_text(encoding="utf-8"); self.assertTrue(audit.source_contract(text)["FIT_only_response_and_six_locks_before_DEV_source"])
        with self.assertRaises(Exception): audit.source_contract(text.replace('fit["target"]', 'tune["target"]'))

    def test_analytic_RMS_uncentered_positive_ridge_and_near_rank_cutoff(self):
        p = np.zeros((2, 40, 1), np.float32); u = np.full((2, 48, 1), 3., np.float32); d = np.zeros((7, 1), np.float32); d[0] = 2.; baseline = np.zeros((2, 48, 1), np.float32); target = np.ones_like(baseline); observed = np.ones_like(baseline, bool)
        fit = audit.independent_fit(p, u, d, baseline, target, observed, "SCALE_ONLY"); np.testing.assert_allclose(fit["scales"], [6.], atol=1e-13, rtol=0)
        for solution in fit["solutions"]: np.testing.assert_allclose(solution["coefficients"], [1/(6*(1+solution["lambda"]))], atol=1e-13, rtol=1e-13)
        p, u, d, raw, cache = fixture(k=2, c=3); p[:, :, 1] = p[:, :, 0]; p[:, -1] = 0; u[:, :, 1] = u[:, :, 0]; u[0, 0] = 0; u[0, 0, 1] = 1e-13; d[1::2] = d[0::2]
        small = audit.independent_fit(p, u, d, cache["prediction"], raw["target"], raw["observed"], "SCALE_ONLY"); self.assertEqual(small["rank"], 1)
        u[0, 0, 1] = 1e-5; large = audit.independent_fit(p, u, d, cache["prediction"], raw["target"], raw["observed"], "SCALE_ONLY"); self.assertEqual(large["rank"], 2)

    def test_complete_synthetic_saved_audit_and_coefficient_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            binding, result, pins = full_fixture(Path(folder))
            with patch.multiple(audit, **pins), patch.object(audit.prior, "check_raw_split", return_value=None): report = audit.audit_result(binding["path"], SimpleNamespace(check=lambda: None))
            self.assertEqual(report["status"], "PASS"); totals = report["totals"]; self.assertEqual([totals[k] for k in ("fit_conditions_checked", "actual_SVD_fits_checked", "ridge_candidates_checked", "identity_candidates_checked", "FIT_TUNE_prediction_evaluations_checked", "selected_DEV_prediction_evaluations_checked")], [6, 6, 42, 6, 96, 6])
            row = report["states"]["RECON_CP0"]["SCALE_BIAS"]; self.assertEqual(len(row["candidates"]), 8); self.assertEqual(len(row["candidates"][1]["a_FP32"]), 4); self.assertEqual(len(row["independently_recomputed_fit"]["column_scales_FP64"]), 8); json.dumps(report, allow_nan=False)
            path = Path(result["states"]["RECON_CP0"]["families"]["SCALE_ONLY"]["candidates"][1]["coefficient_artifact"]["path"]); path.write_bytes(b"tampered")
            with patch.multiple(audit, **pins), patch.object(audit.prior, "check_raw_split", return_value=None), self.assertRaisesRegex(Exception, "SHA_MISMATCH"): audit.audit_result(binding["path"], SimpleNamespace(check=lambda: None))

    def test_failed_family_has_resolved_lock_and_remains_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            binding, result, pins = full_fixture(Path(folder)); state, family = audit.STATES[-1], audit.FAMILIES[-1]; previous = result["states"][state]["families"][family]
            row = {"status": "FAILED_PATH_PRESERVED", "state": state, "family": family, "candidates": [], "failure": {"type": "ValueError", "message": "synthetic SVD failure"}}
            lock = put(Path(previous["selection_lock"]["path"]), {**row, "DEV_opened_before_lock": False, "sealed_unix": 0.}); row["selection_lock"] = lock; result["states"][state]["families"][family] = row
            registry_path = Path(result["selection_registry"]["path"]); registry = json.loads(registry_path.read_text()); registry["conditions"][-1].update(status=row["status"], selection_lock=lock); result["selection_registry"] = put(registry_path, registry)
            result.update(status="PARTIAL_COMPLETED", fit_conditions_completed=5, SVD_fits_completed=5, ridge_candidates_generated=35, identity_comparisons_generated=5, candidate_FIT_TUNE_evaluations_completed=80, selected_DEV_evaluations_completed=5, failed_paths=[{"state": state, "family": family}]); binding = put(Path(binding["path"]), result); pins["RESULT_SHA"] = binding["sha256"]
            receipt = {k: v for k, v in result["resource_terminal"].items() if k not in ("path", "sha256")}; receipt["status"] = "PARTIAL_COMPLETED"; result["resource_terminal"] = {**put(Path(result["resource_terminal"]["path"]), receipt), **receipt}; binding = put(Path(binding["path"]), result); pins["RESULT_SHA"] = binding["sha256"]
            with patch.multiple(audit, **pins), patch.object(audit.prior, "check_raw_split", return_value=None): report = audit.audit_result(binding["path"], SimpleNamespace(check=lambda: None))
            self.assertEqual(report["status"], "PARTIAL_AUDITED_NO_FULL_PASS"); self.assertEqual(report["totals"]["selection_locks_checked"], 6); self.assertEqual(report["totals"]["fit_conditions_checked"], 5); self.assertEqual(len(report["partial_paths"]), 1)

    def test_receipt_canonical_process_CPU_and_file_authorization_chain_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            _, result, _ = full_fixture(Path(folder)); receipt = result["resource_terminal"]; self.assertEqual(len(audit.receipt_check(receipt, "COMPLETE", lambda: None)), 3)
            wrong = copy.deepcopy(receipt); wrong["cpu_seconds"] = wrong.pop("process_cpu_seconds")
            with self.assertRaisesRegex(Exception, "CANONICAL_PROCESS_CPU"): audit.receipt_check(wrong, "COMPLETE", lambda: None)
            wrong = copy.deepcopy(receipt); wrong["wall_seconds"] += 1
            with self.assertRaisesRegex(Exception, "RECEIPT_CONTENT"): audit.receipt_check(wrong, "COMPLETE", lambda: None)
            Path(receipt["authorization"]["path"]).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "JSON_FILE_SHA"): audit.receipt_check(receipt, "COMPLETE", lambda: None)


if __name__ == "__main__": unittest.main()
