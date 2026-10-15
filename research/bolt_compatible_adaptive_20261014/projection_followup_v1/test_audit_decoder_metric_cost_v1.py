"""Independent timer, physical return, source and decoder-binding fixtures."""
import copy
import hashlib
from pathlib import Path
import statistics
import tempfile
import unittest

import numpy as np

import audit_decoder_metric_cost_v1 as audit

audit.initialize_numpy()


def metadata(route="PCA_ENCODER_PINV_BOLT"):
    return {"projection_route": "PCA", "decoder_variant": "ENCODER_PINV", "shared_resident_model": True,
            "reference_prediction_array": "ENCODER_PINV_BOLT_prediction",
            "deployed_decoder_array": "ENCODER_PINV_decoder_deployed_FP32", "E_F_frozen": True,
            "D_frozen": True, "encoder_decoder_refit_in_cost": False}


def timer_fixture():
    memory = {"resident_rss_bytes": 40, "peak_allocated_vram_bytes": 20, "peak_reserved_vram_bytes": 30}
    sample = {"perf_start_ns": 1_000_000_000, "perf_end_ns": 3_000_000_000,
              "unix_start_ns": 5_000_000_000, "unix_end_ns": 7_000_000_000,
              "seconds": 2., "parity": {}, "memory": memory}
    row = {"seconds": [2.], "parity": [{}], "memory": [memory]}
    return sample, row


class IndependentDecoderCostFixtures(unittest.TestCase):
    def test_five_routes_all_use_Bolt_and_first_five_positions_are_balanced(self):
        schedule = audit.independent_schedule()
        self.assertEqual(len(schedule), 6)
        self.assertTrue(all(route.endswith("_BOLT") for route in audit.ROUTES))
        for route in audit.ROUTES:
            self.assertEqual(sorted(order.index(route) for order in schedule[:5]), list(range(5)))
            counts = [sum(order.index(route) == position for order in schedule) for position in range(5)]
            self.assertEqual(sorted(counts), [1, 1, 1, 1, 2])

    def test_source_contract_allows_only_route_literal_and_rejects_changed_timer(self):
        original = 'ROUTES=("RAW_BOLT","PCA_BOLT")\ndef clock():\n return 1\n'
        current = 'ROUTES=' + repr(audit.ROUTES) + '\ndef clock():\n return 1\n'
        audit.timing_copy_contract(original, current)
        with self.assertRaisesRegex(audit.prior.AuditFailure, "ONLY_ROUTE_NAMES"):
            audit.timing_copy_contract(original, current.replace("return 1", "return 2"))

    def test_unknown_or_native_route_is_rejected_before_scientific_payloads(self):
        original = 'ROUTES=("RAW_BOLT",)\n'
        with self.assertRaisesRegex(audit.prior.AuditFailure, "AUTHENTIC_BOLT"):
            audit.timing_copy_contract(original, 'ROUTES=("RAW_BOLT","PCA_NATIVE")\n')

    def test_perf_elapsed_and_row_seconds_have_independent_exact_arithmetic(self):
        sample, row = timer_fixture()
        audit.recorded_timer(sample, row, 0, 0.)
        sample["perf_end_ns"] += 100
        with self.assertRaisesRegex(audit.prior.AuditFailure, "perf_counter"):
            audit.recorded_timer(sample, row, 0, 0.)

    def test_timer_before_seal_or_invalid_peak_counter_is_rejected(self):
        sample, row = timer_fixture()
        with self.assertRaisesRegex(audit.prior.AuditFailure, "SELECTION_LOCK"):
            audit.recorded_timer(sample, row, 0, 8.)
        sample["memory"]["peak_allocated_vram_bytes"] = 31
        with self.assertRaisesRegex(audit.prior.AuditFailure, "MEMORY_COUNTERS"):
            audit.recorded_timer(sample, row, 0, 0.)

    def test_paired_ratio_median_differs_from_ratio_of_unpaired_medians(self):
        seconds, raw = [1., 2., 3., 40., 50., 60.], [1., 1., 1., 10., 10., 100.]
        ratios = [a / b for a, b in zip(seconds, raw)]
        row = {"seconds": seconds, "batch": 8, "paired_RAW_seconds": raw, "paired_RAW_ratios": ratios,
               "median_seconds": statistics.median(seconds), "median_seconds_per_window": statistics.median(seconds) / 8,
               "paired_RAW_median_ratio": statistics.median(ratios), "repeats": 6, "warmup_each_route": 2}
        audit.prior.audit_recorded_times(row, {"seconds": raw, "batch": 8})
        row["paired_RAW_median_ratio"] = statistics.median(seconds) / statistics.median(raw)
        with self.assertRaisesRegex(audit.prior.AuditFailure, "median_ratio"):
            audit.prior.audit_recorded_times(row, {"seconds": raw, "batch": 8})

    def test_causal_forward_fill_resets_each_window_to_FIT_mean(self):
        physical = np.full((2, 512, 1), np.nan, np.float32)
        mask = np.zeros_like(physical, bool)
        physical[0, 0, 0], mask[0, 0, 0] = 14., True
        physical[1, 10, 0], mask[1, 10, 0] = 18., True
        normalized = audit.prior.causal_physical_preprocess(physical, mask, np.array([10.], np.float32), np.array([2.], np.float32))
        self.assertEqual(normalized[0, -1, 0], 2.)
        self.assertEqual(normalized[1, 9, 0], 0.)
        self.assertEqual(normalized[1, 10, 0], 4.)

    def test_wrong_physical_units_cannot_pass_saved_output_parity(self):
        normalized = np.array([[[1., -2.]]], np.float32)
        mean, std = np.array([100., 20.], np.float32), np.array([3., 8.], np.float32)
        physical = (normalized * std + mean).astype(np.float32)
        record = {"pass": True, "atol": 2e-3, "rtol": 2e-5, "max_abs": 0., "rmse": 0.}
        audit.prior.compare_parity_record(record, physical, physical, "physical", atol=2e-3, rtol=2e-5)
        with self.assertRaisesRegex(audit.prior.AuditFailure, "NUMERIC_MISMATCH"):
            audit.prior.compare_parity_record(record, (normalized + mean).astype(np.float32), physical, "physical", atol=2e-3, rtol=2e-5)

    def test_matched_scalar_predictions_cannot_hide_wrong_decoder_route(self):
        valid = metadata()
        audit.validate_route_metadata("PCA_ENCODER_PINV_BOLT", valid, copy.deepcopy(valid))
        invalid = {**valid, "reference_prediction_array": "ORIGINAL_PAST_LS_BOLT_prediction"}
        with self.assertRaisesRegex(audit.prior.AuditFailure, "CALLBACK_METADATA"):
            audit.validate_route_metadata("PCA_ENCODER_PINV_BOLT", invalid, copy.deepcopy(invalid))

    def test_canonical_bridge_requires_full_matching_path_hash_and_member_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.npz"
            np.savez(path, x=np.array([3.], np.float32))
            record = {"path": str(path), "sha256": audit.prior.file_sha(path), "arrays": {"x": audit.array_record(np.array([3.], np.float32))}}
            output = audit.prior.bind_canonical_npz({"path": str(path), "sha256": record["sha256"]}, record)
            self.assertEqual(output["arrays"], record["arrays"])
            wrong = {"path": str(path), "sha256": "0" * 64}
            with self.assertRaises(audit.prior.AuditFailure):
                audit.prior.bind_canonical_npz(wrong, record)

    def test_byte_binding_detects_signed_zero_and_origin_middle_changes(self):
        self.assertTrue(np.array_equal(np.array([0.], np.float32), np.array([-0.], np.float32)))
        with self.assertRaisesRegex(audit.prior.AuditFailure, "BYTE_EXACT"):
            audit.exact(np.array([0.], np.float32), np.array([-0.], np.float32), "physical_signed_zero")
        with self.assertRaisesRegex(audit.prior.AuditFailure, "BYTE_EXACT"):
            audit.exact(np.array([1, 3, 7], np.int64), np.array([1, 4, 7], np.int64), "same_endpoint_wrong_origin")


if __name__ == "__main__":
    unittest.main()
