"""Synthetic physical/rank/timer metadata fixtures; no model or actual assets."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import audit_local_past_decoder_cost_v1 as audit
from test_audit_local_past_decoder_probe_v1 import integrated_case

audit.initialize_numpy()


def source_texts():
    return [(audit.HERE / name).read_text(encoding="utf-8-sig") for name in
            ("whole_fit_target_cost_v1.py", "whole_local_past_decoder_cost_v1.py", "run_local_past_decoder_cost_v1.py", "local_past_decoder_v1.py")]


def parity(value, expected, atol=1e-5, rtol=2e-6):
    error = value.astype(np.float64) - expected.astype(np.float64)
    return {"pass": bool(np.all(np.abs(error) <= atol + rtol * np.abs(expected))), "max_abs": float(np.abs(error).max()),
            "rmse": float(np.sqrt(np.square(error).mean())), "atol": atol, "rtol": rtol}


def physical_case():
    single, single_arrays, single_state, single_values, _, _, _ = integrated_case()
    values = {key: np.repeat(value, 8, axis=0) if value.ndim > 1 or key == "origins" else value.copy() for key, value in single_values.items()}
    values["origins"] = np.arange(512, 520, dtype=np.int64)
    values["mean"] = np.zeros(2, np.float32)
    values["std"] = np.ones(2, np.float32)
    z = np.repeat(single_arrays["latent_history"], 8, axis=0)
    metadata = copy.deepcopy(single["local_fit_metadata"])
    metadata["windows"] = 8
    for key in ("true_observed_counts", "centered_slope_ranks", "affine_model_ranks_with_analytic_intercept", "retained_condition_number", "observed_reconstruction_MSE_real_fit", "observed_latent_means_per_C", "observed_raw_C_means", "singularvalues_observed_centered_Z", "rank_deficient_slope_channels", "observed_count_below_two", "no_observation_current_C_fallback", "observed_reconstruction_MSE_valid", "retained_condition_number_valid", "decoder_Frobenius_norm_per_window", "decoder_operator_norm_per_window", "max_abs_decoder_per_window"):
        metadata[key] = np.repeat(np.asarray(metadata[key]), 8, axis=0).tolist()
    state = {key: np.repeat(value, 8, axis=0) for key, value in single_state.items() if key != "metadata_json"}
    metadata["fit_input_bindings"] = {"history": audit.array_record(values["history"]), "history_observed": audit.array_record(values["history_observed"]), "cached_latent_history": audit.array_record(z)}
    metadata["state_arrays"] = {name: audit.array_record(state[key]) for name, key in (("decoder_fit64", "decoder_fit_FP64"), ("decoder32", "decoder_deployed_FP32"), ("bias_fit64", "decoder_bias_fit_FP64"), ("bias32", "decoder_bias_FP32"))}
    state["metadata_json"] = np.array(json.dumps(metadata, allow_nan=False))
    arrays = {**state, "preprocessed_history": values["history"].copy(), "latent_history": z.copy(), "history_observed": values["history_observed"].copy(), "origins": values["origins"].copy()}
    physical = {"history_physical": values["history"].copy(), "history_observed": values["history_observed"].copy(), "origins": values["origins"].copy()}
    encoder_state = {"center": np.array([3., 0.]), "encoder": np.array([[.5], [0.]])}
    sources = {name: {"path": "synthetic_" + name, "sha256": "fixture"} for name in ("source_local_state", "source_local_forecast", "source_state", "source_BOLT_cache")}
    checks = {key: parity(actual, actual) for key, actual in (("normalized_history", values["history"]), ("normalized_current_C", values["raw_last"]), ("encoded_past", z), ("decoder_FP32", state["decoder_deployed_FP32"]), ("decoder_bias_FP32", state["decoder_bias_FP32"]), ("singularvalues_FP64", np.asarray(metadata["singularvalues_observed_centered_Z"])))}
    decisions = {key: True for key in ("true_observed_counts", "centered_slope_ranks", "affine_model_ranks_with_analytic_intercept", "no_observation_current_C_fallback", "rank_deficient_slope_channels", "observed_count_below_two")}
    entry = {"status": "PASS", **sources, "checks": checks, "exact_decisions": decisions, "history_observed_byte_exact": True,
             "singular_relative_cut": 1e-4, "origins": values["origins"].tolist(), "rank_acceptance_by_output_tolerance_only": False,
             "coefficient_tolerance_loosened": False, "physical_input_rounding_can_change_local_solver_or_output": True, "fitted_local_metadata": metadata}
    return entry, arrays, physical, values, encoder_state, {"latent_history": z}, state, metadata, sources


def save_arrays(path, values):
    np.savez_compressed(path, **values)
    return {"path": str(path.resolve()), "sha256": audit.prior.file_sha(path), "arrays": {key: audit.array_record(value) for key, value in values.items()}}


def fake_complete_cost(folder):
    _, _, physical, dev, _, _, _, _, _ = physical_case()
    dev["history"] = dev["history"].astype(np.float32)
    predictions = {route: dev["target"].copy() for route in audit.ROUTES}
    raw_meta = {"source_inputs": {"path": "synthetic", "sha256": "fixture"}, "reference_prediction_array": "raw_forecast", "shared_resident_frozen_F": True, "local_fit": False, "native_head_loaded_or_timed": False}
    metadata = {"RAW_BOLT": raw_meta}
    for route in audit.ROUTES[1:]:
        local = route.endswith("LOCAL_BOLT")
        metadata[route] = {"local_fit": local, "local_fit_uses_current_known_past_and_true_observed_mask": local,
            "local_LS_and_implemented_diagnostics_inside_timer": local, "cached_local_D_used_for_timing": False,
            "new_E_F_or_head_training": False, "strict_K_only_compression": False, "past_C_condition_metadata_added": local,
            "same_decoder_side_information_as_static": not local, "singular_relative_cut": 1e-4 if local else None,
            "reference_prediction_array": "BOLT_local_prediction" if local else "BOLT_static_prediction", "shared_resident_frozen_F": True, "native_head_loaded_or_timed": False}
    schedule, rows = audit.independent_schedule(), []
    for batch in (1, 8):
        raw_seconds = [.01 * (repeat + 1) for repeat in range(6)]
        for route_index, route in enumerate(audit.ROUTES):
            norm = predictions[route][:batch]
            phys = (norm * dev["std"] + dev["mean"]).astype(np.float32)
            origins = dev["origins"][:batch]
            reference = save_arrays(folder / (str(batch) + route + "REF.npz"), {"normalized": norm, "physical": phys, "scientific_normalized": norm, "origins": origins})
            seconds = [raw * (route_index + 1) for raw in raw_seconds]
            samples, memories, parities = [], [], []
            positions = [block.index(route) for block in schedule]
            for repeat, position in enumerate(positions):
                timestamp = (batch * 1000 + repeat * 10 + position) * 1_000_000_000
                duration = int(round(seconds[repeat] * 1e9))
                sample_path = folder / (str(batch) + route + str(repeat) + ".npz")
                artifact = save_arrays(sample_path, {"normalized": norm, "physical": phys, "origins": origins})
                memory = {"resident_rss_bytes": 100, "peak_allocated_vram_bytes": 50, "peak_reserved_vram_bytes": 60}
                checked = {"normalized": parity(norm, norm), "physical": parity(phys, phys, 2e-3, 2e-5), "scientific_normalized": parity(norm, norm), "scientific_physical": parity(phys, phys, 2e-3, 2e-5)}
                sample = {"dataset": "fixture", "route": route, "batch": batch, "repeat": repeat, "block": repeat, "position": position,
                    "seconds": seconds[repeat], "perf_start_ns": timestamp, "perf_end_ns": timestamp + duration,
                    "unix_start_ns": timestamp, "unix_end_ns": timestamp + duration, "parity": checked, "memory": memory, "sample_artifact": artifact}
                sample_path.with_suffix(".json").write_text(json.dumps(sample), encoding="utf-8")
                samples.append(sample); memories.append(memory); parities.append(checked)
            timed = save_arrays(folder / (str(batch) + route + "TIMED.npz"), {"normalized": np.repeat(norm[None], 6, axis=0), "physical": np.repeat(phys[None], 6, axis=0), "origins": origins, "block": np.arange(6, dtype=np.int64), "position": np.array(positions, np.int64)})
            rows.append({"dataset": "fixture", "route": route, "batch": batch, "status": "COMPLETE", "warmup_each_route": 2, "repeats": 6,
                "seconds": seconds, "median_seconds": float(np.median(seconds)), "median_seconds_per_window": float(np.median(seconds))/batch,
                "paired_RAW_seconds": raw_seconds, "paired_RAW_ratios": [float(route_index + 1)]*6, "paired_RAW_median_ratio": float(route_index+1),
                "positions": positions, "origins": origins.tolist(), "parity": parities, "memory": memories, "samples": samples,
                "selected_metadata": metadata[route], "timed_outputs": timed, "reference": reference,
                "memory_counter_reset_inside_module": False, "stage_decomposition": False, "native_head_loaded_or_timed": False})
    cost = {"dataset": "fixture", "status": "COMPLETE", "routes": list(audit.ROUTES), "expected_cells_this_dataset": 10, "expected_cells_two_datasets": 20,
        "rows": rows, "warmup_each_route": 2, "repeats": 6, "new_training_updates": 0, "encoder_refit_in_cost": False,
        "local_decoder_fit_inside_timed_callback": True, "implemented_local_fit_diagnostics_included": True, "protected_TEST_accessed": False,
        "independent_holdout": False, "recorded_timer_samples": 60, "attempted_timer_calls": 60, "order_seed": 20261015, "schedule": schedule,
        "source_metadata": metadata, "input_array_metadata": {key: audit.array_record(value) for key,value in (("physical_past",physical["history_physical"]),("history_observed",physical["history_observed"]),("FIT_mean",dev["mean"]),("FIT_std",dev["std"]),("origins",dev["origins"]))},
        "source_lineage": {"whole_cost_source_sha256": audit.prior.file_sha(audit.HERE/"whole_local_past_decoder_cost_v1.py"),
            "copied_preprocess_source": str(audit.BASE/"cost_readout_v1.py"), "copied_preprocess_source_sha256": audit.prior.file_sha(audit.BASE/"cost_readout_v1.py"), "copied_functions": ["fill_vectorized","preprocess"], "legacy_scientific_runtime_imported": False}}
    return cost, physical, dev, predictions, metadata


class IndependentLocalWholeCostFixtures(unittest.TestCase):
    def test_actual_sources_have_LS_full_diagnostics_inside_timer(self):
        contract = audit.source_contract(*source_texts())
        self.assertTrue(contract["local_LS_and_registered_full_diagnostics_inside_each_callback"])
        self.assertFalse(contract["cached_D_in_timed_pipeline"])

    def test_wrong_seed_prefix_is_rejected_before_numeric_IO(self):
        texts=source_texts();texts[1]=texts[1].replace("ORDER_SEED = 20261015","ORDER_SEED = 24261015")
        with self.assertRaisesRegex(ValueError,"SEED"):
            audit.source_contract(*texts)

    def test_fit_outside_callback_or_cached_D_is_rejected(self):
        texts=source_texts();texts[2]=texts[2].replace("local = fit_local_past_decoder(history, history_observed, z)","local = cached_local_state")
        with self.assertRaisesRegex(ValueError,"EACH_TIMED_CALL"):
            audit.source_contract(*texts)

    def test_removed_diagnostic_norm_or_hash_is_rejected(self):
        texts=source_texts();texts[3]=texts[3].replace("np.linalg.svd(matrix, compute_uv=False)","np.zeros(1)")
        with self.assertRaisesRegex(ValueError,"DIAGNOSTICS"):
            audit.source_contract(*texts)

    def test_missing_explicit_true_past_mask_is_rejected(self):
        texts=source_texts();texts[1]=texts[1].replace("predictor(normalized_history, observed)","predictor(normalized_history)")
        with self.assertRaisesRegex(ValueError,"MASK"):
            audit.source_contract(*texts)

    def test_five_rotations_balance_positions_sixth_is_extra(self):
        schedule=audit.independent_schedule()
        self.assertEqual(len(schedule),6)
        for route in audit.ROUTES:
            self.assertEqual(sorted(row.index(route) for row in schedule[:5]),list(range(5)))
            self.assertEqual(len([row.index(route) for row in schedule]),6)

    def test_physical_first8_full_independent_state_and_decisions(self):
        result=audit.independent_physical_precheck(*physical_case())
        self.assertEqual(result["state_checks"]["window_matrix_fits_checked"],8)
        self.assertEqual(result["state_checks"]["channel_slots_checked"],16)
        self.assertFalse(result["rank_or_fallback_output_tolerance_acceptance_used"])
        json.loads(json.dumps(result,allow_nan=False))

    def test_output_and_coefficients_matching_does_not_hide_rank_change(self):
        args=list(physical_case());args[7]=copy.deepcopy(args[7]);args[7]["centered_slope_ranks"][0][0]=0
        with self.assertRaisesRegex(ValueError,"BYTE"):
            audit.independent_physical_precheck(*args)

    def test_observed_mask_rounding_parity_cannot_replace_exactmask(self):
        args=list(physical_case());args[3]["history_observed"]=args[3]["history_observed"].copy();args[3]["history_observed"][0,0,0]=False
        with self.assertRaisesRegex(ValueError,"BYTE"):
            audit.independent_physical_precheck(*args)

    def test_full_synthetic_10cells_60samples_savedarrays_timer_arithmetic(self):
        with tempfile.TemporaryDirectory(prefix="local_whole_audit_fixture_") as temporary:
            args=fake_complete_cost(Path(temporary))
            result=audit.audit_cost_dataset(*args,lambda:None,sealed_unix=0.)
            self.assertEqual(result["cells_checked"],10);self.assertEqual(result["recorded_timer_samples_checked"],60)
            self.assertFalse(result["clock_memory_or_F_forward_independently_remeasured"])
            self.assertEqual(result["rows"][3]["paired_RAW_median_ratio"],4.)
            json.loads(json.dumps(result,allow_nan=False))

    def test_wrong_recorded_seed_and_duplicate_cell_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="local_whole_audit_fixture_") as temporary:
            args=fake_complete_cost(Path(temporary));args[0]["order_seed"]=24261015
            with self.assertRaisesRegex(ValueError,"SCHEDULE"):
                audit.audit_cost_dataset(*args,lambda:None,sealed_unix=0.)
            args[0]["order_seed"]=20261015;args[0]["rows"][-1]=args[0]["rows"][0]
            with self.assertRaisesRegex(ValueError,"DUPLICATE"):
                audit.audit_cost_dataset(*args,lambda:None,sealed_unix=0.)

    def test_changed_median_and_sample_counter_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="local_whole_audit_fixture_") as temporary:
            args=fake_complete_cost(Path(temporary));args[0]["recorded_timer_samples"]=59
            with self.assertRaisesRegex(ValueError,"SAMPLE_COUNTS"):
                audit.audit_cost_dataset(*args,lambda:None,sealed_unix=0.)
            args[0]["recorded_timer_samples"]=60;args[0]["rows"][0]["median_seconds"]+=.01
            with self.assertRaisesRegex(ValueError,"NUMERIC_MISMATCH"):
                audit.audit_cost_dataset(*args,lambda:None,sealed_unix=0.)

    def test_saved_sample_mutation_is_rejected_by_manifest(self):
        with tempfile.TemporaryDirectory(prefix="local_whole_audit_fixture_") as temporary:
            args=fake_complete_cost(Path(temporary));sample=Path(args[0]["rows"][0]["samples"][0]["sample_artifact"]["path"])
            with sample.open("ab") as handle:handle.write(b"fixture-mutation")
            with self.assertRaisesRegex(ValueError,"SHA"):
                audit.audit_cost_dataset(*args,lambda:None,sealed_unix=0.)


if __name__=="__main__":
    unittest.main()
