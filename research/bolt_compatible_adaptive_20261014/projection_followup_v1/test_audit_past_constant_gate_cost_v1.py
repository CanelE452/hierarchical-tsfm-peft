"""Independent Latin6, exact decisions, physical returns and timed-source fixtures."""
import copy
import inspect
import unittest

import numpy as np

import audit_past_constant_gate_cost_v1 as audit

audit.initialize_numpy()


def physical_fixture():
    history = np.ones((8, 512, 2), np.float32)
    mask = np.ones_like(history, bool)
    dev = {"history": history.copy(), "history_observed": mask.copy(), "raw_last": history[:, -1].copy(),
           "mean": np.zeros(2, np.float32), "std": np.ones(2, np.float32), "origins": np.arange(8, dtype=np.int64) + 512}
    physical = {"history_physical": history.copy(), "history_observed": mask.copy()}
    refs = {"keep_forecast": np.zeros((8, 2), bool), "exact_flat": np.ones((8, 2), bool),
            "true_observed_counts": np.full((8, 2), 512, np.int64), "raw_last": dev["raw_last"].copy()}
    physical_record, reference_record = {"path": "physical.npz", "sha256": "1"}, {"path": "CPU.npz", "sha256": "2"}
    parity = {"pass": True, "max_abs": 0., "rmse": 0., "atol": 1e-5, "rtol": 2e-6}
    entry = {"status": "PASS", "physical_source": physical_record, "CPU_gate_source": reference_record,
             "origins": dev["origins"].tolist(), "all_three_gate_arrays_byte_exact": True,
             "tolerance_parity_used_to_accept_gate_decision": False, "history_observed": audit.array_record(mask),
             "preprocessed_history": audit.array_record(history),
             "actual_physical_gate_arrays": {key: audit.array_record(value) for key, value in refs.items() if key != "raw_last"},
             "CPU_reference_gate_arrays": {key: audit.array_record(value) for key, value in refs.items() if key != "raw_last"},
             "normalized_history_parity": copy.deepcopy(parity), "normalized_current_C_parity": copy.deepcopy(parity)}
    return entry, physical, dev, refs, physical_record, reference_record


def route_metadata(route):
    gated = route.endswith("_GATED")
    base = route[:-6] if gated else route
    return {"base_method": base, "gated": gated, "reference_prediction_array": base + ("_gated_prediction" if gated else "_ungated_prediction"),
            "shared_resident_frozen_F": True, "native_head_loaded_or_timed": False,
            "history_observed_explicitly_passed": True, "past_C_metadata_added": gated,
            "same_decoder_side_information_as_ungated": not gated, "new_fit": False}


class IndependentWholeConstantGateFixtures(unittest.TestCase):
    def test_all_six_ROUTES_receive_all_six_positions_exactly_once(self):
        schedule = audit.independent_schedule()
        self.assertEqual(len(schedule), 6)
        for route in audit.ROUTES:
            self.assertEqual(sorted(order.index(route) for order in schedule), list(range(6)))
        self.assertEqual(sum(len(order) for order in schedule), 36)

    def test_six_routes_include_fair_gated_RAW_and_no_native(self):
        self.assertIn("RAW_BOLT_GATED", audit.ROUTES)
        self.assertFalse(any("NATIVE" in route for route in audit.ROUTES))
        for route in audit.ROUTES:
            entry = route_metadata(route)
            audit.validate_route_metadata(route, entry, dict(entry))

    def test_wrong_CPU_reference_variant_cannot_be_accepted(self):
        entry = route_metadata("PCA_PINV_BOLT_GATED")
        entry["reference_prediction_array"] = "PCA_PINV_BOLT_ungated_prediction"
        with self.assertRaisesRegex(audit.prior.AuditFailure, "AUTHENTIC_F_ONLY"):
            audit.validate_route_metadata("PCA_PINV_BOLT_GATED", entry, dict(entry))

    def test_explicit_true_past_mask_is_required_by_timed_callback_metadata(self):
        entry = route_metadata("RAW_BOLT_GATED")
        entry["history_observed_explicitly_passed"] = False
        with self.assertRaisesRegex(audit.prior.AuditFailure, "AUTHENTIC_F_ONLY"):
            audit.validate_route_metadata("RAW_BOLT_GATED", entry, dict(entry))

    def test_original_perf_span_and_each_inside_gate_call_are_verified_from_AST(self):
        original = (audit.HERE / "whole_cost_v1.py").read_text(encoding="utf-8-sig")
        helper = (audit.HERE / "whole_past_constant_gate_cost_v1.py").read_text(encoding="utf-8-sig")
        runner = (audit.HERE / "run_past_constant_gate_cost_v1.py").read_text(encoding="utf-8-sig")
        self.assertTrue(audit.source_contract(original, helper, runner)["gate_computation_inside_each_timed_callback"])
        with self.assertRaisesRegex(audit.prior.AuditFailure, "OBSERVED_MASK_EXPLICIT"):
            audit.source_contract(original, helper.replace("predictor(normalized_history, observed)", "predictor(normalized_history)"), runner)
        with self.assertRaisesRegex(audit.prior.AuditFailure, "PAST_ONLY_SIGNATURE"):
            audit.source_contract(original, helper, runner.replace("def predict(history, history_observed):", "def predict(history):"))

    def test_exact_physical_preflight_checks_all_three_gate_arrays(self):
        values = physical_fixture()
        result = audit.independent_physical_gate_precheck(*values)
        self.assertTrue(result["metadata"]["first_B1_B8_gate_arrays_byte_exact"])
        self.assertEqual(result["metadata"]["physical_gate_eligible_window_channel_pairs_first8"], 16)

    def test_one_ulp_passes_numeric_input_parity_but_fails_exact_gate_precheck(self):
        values = physical_fixture()
        entry, physical, dev, refs, physical_record, reference_record = values
        physical["history_physical"][0, 10, 0] = np.nextafter(np.float32(1.), np.float32(2.))
        self.assertLess(float(np.max(np.abs(physical["history_physical"] - dev["history"]))), 1e-5)
        with self.assertRaisesRegex(audit.prior.AuditFailure, "decision_no_tolerance"):
            audit.independent_physical_gate_precheck(*values)

    def test_wrong_observed_counts_fail_even_when_past_values_unchanged(self):
        values = physical_fixture()
        values[1]["history_observed"][0, 20, 0] = False
        with self.assertRaisesRegex(audit.prior.AuditFailure, "observed_counts"):
            audit.independent_physical_gate_precheck(*values)

    def test_gate_preflight_metadata_cannot_use_threshold_tolerance(self):
        values = physical_fixture()
        values[0]["tolerance_parity_used_to_accept_gate_decision"] = True
        with self.assertRaisesRegex(audit.prior.AuditFailure, "GATE_IDENTITY_SCOPE"):
            audit.independent_physical_gate_precheck(*values)

    def test_recorded_timer_arithmetic_is_not_accepting_stale_rounded_seconds(self):
        memory = {"resident_rss_bytes": 3, "peak_allocated_vram_bytes": 1, "peak_reserved_vram_bytes": 2}
        sample = {"perf_start_ns": 1_000_000_000, "perf_end_ns": 1_025_000_001,
                  "unix_start_ns": 5_000_000_000, "unix_end_ns": 5_025_000_001,
                  "seconds": .025, "parity": {}, "memory": memory}
        row = {"seconds": [.025], "parity": [{}], "memory": [memory]}
        with self.assertRaisesRegex(audit.prior.AuditFailure, "perf_counter"):
            audit.recorded_timer(sample, row, 0, 0.)


if __name__ == "__main__":
    unittest.main()
