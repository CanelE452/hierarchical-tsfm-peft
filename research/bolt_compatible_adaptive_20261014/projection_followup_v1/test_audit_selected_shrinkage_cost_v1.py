"""Independent augmented-LS and saved-clock fixtures only; no model/F replay."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import audit_selected_shrinkage_cost_v1 as audit
from test_audit_local_decoder_shrinkage_v1 import state_fixture
from test_audit_local_past_decoder_cost_v1 import fake_complete_cost as old_cost_fixture

audit.initialize_numpy()


def source_texts():
    return [(audit.HERE / name).read_text(encoding="utf-8-sig") for name in
        ("whole_local_past_decoder_cost_v1.py", "whole_selected_shrinkage_cost_v1.py", "run_selected_shrinkage_cost_v1.py", "local_decoder_shrinkage_v1.py")]


def parity(value, expected, atol=1e-5, rtol=2e-6):
    error = np.asarray(value).astype(np.float64) - np.asarray(expected).astype(np.float64)
    return {"pass": bool(np.all(np.abs(error) <= atol + rtol * np.abs(expected))), "max_abs": float(np.abs(error).max()),
        "rmse": float(np.sqrt(np.square(error).mean())), "atol": atol, "rtol": rtol}


def physical_case():
    z = np.tile(np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], np.float32), (128, 1))[None]
    z = np.repeat(z, 8, axis=0)
    x = np.concatenate((z, z[:, :, :1] + 2 * z[:, :, 1:]), axis=2).astype(np.float32)
    observed = np.ones(x.shape, bool)
    d0 = np.array([[.5, 0., .5], [0., .5, 1.]], np.float32)
    rho = .1
    metadata, science_state = state_fixture(x, observed, z, d0, rho)
    state = {"center": np.zeros(3), "encoder": np.array([[1., 0.], [0., 1.], [0., 0.]]), "decoder": d0.astype(np.float64)}
    values = {"history": x, "history_observed": observed, "mean": np.zeros(3, np.float32), "std": np.ones(3, np.float32),
        "raw_last": x[:, -1], "origins": np.arange(512, 520, dtype=np.int64)}
    physical = {"history_physical": x, "history_observed": observed, "origins": values["origins"]}
    arrays = {**science_state, "preprocessed_history": x, "latent_history": z, "history_observed": observed,
        "origins": values["origins"], "prior_decoder_FP32": d0}
    sources = {key: {"path": "synthetic", "sha256": "fixture"} for key in
        ("source_shrunk_state", "source_shrunk_forecast", "selection_lock", "source_state", "source_BOLT_cache")}
    sources["selected_candidate"] = "RHO_0.1"
    numerical = {key: parity(value, value) for key, value in (("normalized_history", x), ("normalized_current_C", x[:, -1]),
        ("encoded_past", z), ("decoder_FP32", science_state["decoder_deployed_FP32"]), ("decoder_bias_FP32", science_state["decoder_bias_FP32"]))}
    numerical.update({key: parity(np.asarray(metadata[key]), np.asarray(metadata[key])) for key in
        ("actual_lambda", "local_covariance_trace", "ridge_matrix_condition", "real_stationarity_residual_norm")})
    decisions = {key: True for key in ("true_observed_counts", "zero_trace_prior_slope_retained", "no_observation_prior_with_current_intercept",
        "past_metric_valid", "ridge_matrix_condition_valid", "positive_lambda_solver_decision")}
    entry = {"status": "PASS", **sources, "rho": rho, "lambda_rule": metadata["lambda_rule"],
        "checks": numerical, "exact_decisions": decisions, "history_observed_byte_exact": True, "origins": values["origins"].tolist(),
        "decision_acceptance_by_output_tolerance_only": False, "coefficient_tolerance_loosened": False,
        "fitted_shrunk_metadata": metadata, "deployed_static_prior": audit.array_record(d0)}
    return entry, arrays, physical, values, state, {"latent_history": z}, science_state, metadata, sources, rho


def cost_fixture(folder):
    values = list(old_cost_fixture(folder))
    cost, physical, dev, predictions, metadata = values
    old_routes = list(cost["routes"])
    mapping = dict(zip(old_routes, audit.ROUTES))
    cost["routes"] = list(audit.ROUTES)
    cost["schedule"] = [[mapping[route] for route in row] for row in cost["schedule"]]
    cost["source_lineage"]["whole_cost_source_sha256"] = audit.prior.file_sha(audit.HERE / "whole_selected_shrinkage_cost_v1.py")
    cost["fit_input_binding_hashes_inside_timer"] = True
    predictions = {mapping[route]: value for route, value in predictions.items()}
    metadata = {mapping[route]: value for route, value in metadata.items()}
    for route, value in metadata.items():
        if route == "RAW_BOLT": continue
        shrunk = route.endswith("SHRUNK_BOLT")
        value.pop("local_LS_and_implemented_diagnostics_inside_timer")
        value.pop("singular_relative_cut")
        value.update(local_ridge_and_implemented_diagnostics_inside_timer=shrunk, fit_input_binding_hashes_inside_timer=shrunk,
            new_rho_selection_in_cost=False, reference_prediction_array="BOLT_prediction" if shrunk else "BOLT_static_prediction")
    cost["source_metadata"] = metadata
    for row in cost["rows"]:
        row["route"] = mapping[row["route"]]
        row["selected_metadata"] = metadata[row["route"]]
        for sample in row["samples"]:
            sample["route"] = row["route"]
            Path(sample["sample_artifact"]["path"]).with_suffix(".json").write_text(json.dumps(sample), encoding="utf-8")
    return cost, physical, dev, predictions, metadata


class IndependentSelectedCostTests(unittest.TestCase):
    def test_actual_source_fit_and_hash_boundaries_are_inside_timer(self):
        result = audit.source_contract(*source_texts())
        self.assertTrue(result["selected_ridge_and_registered_full_diagnostics_input_hashes_inside_each_callback"])
        self.assertFalse(result["independent_F_or_timer_replay_performed"])

    def test_cached_D_missing_mask_and_removed_fit_hash_are_rejected(self):
        cases = ((2, "fitted = fit_shrunk_local_decoder(history, history_observed, z, prior32, rho)", "fitted = cached_state"),
            (1, "predictor(normalized_history, observed)", "predictor(normalized_history)"),
            (3, "hashlib.sha256(value.tobytes())", "hashlib.sha256(b'cached')"))
        for index, original, changed in cases:
            texts = source_texts(); self.assertIn(original, texts[index]); texts[index] = texts[index].replace(original, changed)
            with self.subTest(index=index), self.assertRaises(ValueError): audit.source_contract(*texts)

    def test_independent_first8_augmented_LS_coefficients_lambda_bias_and_six_decisions(self):
        result = audit.independent_physical_precheck(*physical_case())
        self.assertEqual(result["state_checks"]["window_fits_checked"], 8)
        self.assertEqual(result["state_checks"]["channel_slots_checked"], 24)
        self.assertEqual(len(result["six_decisions_verified_exact"]), 6)
        json.dumps(result, allow_nan=False)

    def test_same_output_cannot_hide_wrong_rho_lambda_mask_or_prior(self):
        for field in ("rho", "mask", "prior", "lambda"):
            args = list(physical_case())
            if field == "rho": args[-1] = 1.
            elif field == "mask": args[3]["history_observed"] = args[3]["history_observed"].copy(); args[3]["history_observed"][0, 0, 0] = False
            elif field == "prior": args[4]["decoder"][0, 0] += .2
            else: args[7] = copy.deepcopy(args[7]); args[7]["actual_lambda"][0][0] += .1
            with self.subTest(field=field), self.assertRaises(ValueError): audit.independent_physical_precheck(*args)

    def test_saved_coefficient_alteration_is_independently_rejected(self):
        args = list(physical_case()); args[1]["decoder_fit_FP64"] = args[1]["decoder_fit_FP64"].copy()
        args[1]["decoder_fit_FP64"][0, 0, 0] += .1
        with self.assertRaises(ValueError): audit.independent_physical_precheck(*args)

    def test_five_rotation_positions_sixth_extra(self):
        rows = audit.independent_schedule()
        self.assertEqual(len(rows), 6)
        for route in audit.ROUTES: self.assertEqual(sorted(row.index(route) for row in rows[:5]), list(range(5)))

    def test_saved_10cells_60samples_clock_output_parity_without_remeasurement(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = audit.audit_cost_dataset(*cost_fixture(Path(temporary)), lambda: None, sealed_unix=0.)
        self.assertEqual(result["cells_checked"], 10)
        self.assertEqual(result["recorded_timer_samples_checked"], 60)
        self.assertFalse(result["clock_memory_or_F_forward_independently_remeasured"])

    def test_bad_sample_counts_median_and_sixth_positions_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = list(cost_fixture(Path(temporary)))
            args[0]["recorded_timer_samples"] = 59
            with self.assertRaisesRegex(ValueError, "SAMPLE_COUNTS"): audit.audit_cost_dataset(*args, lambda: None, sealed_unix=0.)
            args[0]["recorded_timer_samples"] = 60
            args[0]["rows"][0]["median_seconds"] += .1
            with self.assertRaisesRegex(ValueError, "NUMERIC_MISMATCH"): audit.audit_cost_dataset(*args, lambda: None, sealed_unix=0.)

    def test_saved_output_file_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = cost_fixture(Path(temporary))
            path = Path(args[0]["rows"][0]["samples"][0]["sample_artifact"]["path"])
            with path.open("ab") as handle: handle.write(b"synthetic corruption")
            with self.assertRaisesRegex(ValueError, "SHA"): audit.audit_cost_dataset(*args, lambda: None, sealed_unix=0.)

    def test_incomplete_metadata_does_not_open_scientific_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "incomplete.json"; path.write_text(json.dumps({"status": "RUNNING"}), encoding="utf-8")
            record, value, parents, rules = audit.metadata_gates(path, lambda: None)
            self.assertEqual(value["status"], "RUNNING"); self.assertEqual((parents, rules), ({}, {}))


if __name__ == "__main__": unittest.main()
