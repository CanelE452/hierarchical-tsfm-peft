"""Stdlib synthetic report fixtures; no campaign result or scientific import."""
import contextlib
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name("build_b0_b2_report.py")
if SOURCE.exists():
    spec = importlib.util.spec_from_file_location("prepared_b0_b2_report", SOURCE)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
else:
    builder = None


def fixture_row(family, epoch=0., missing_cost=False):
    width = {"RAW": 32, "DIRECT_G": 0, "B0": 8, "B2": 8}[family]
    registered = 8 if family in ("B0", "B2") else 0
    run = f"postT_cpu_pems08_E1_SUBSET_{family}_k{registered}_s92601_C32_priority_v1"
    fit = dict(status="COMPLETE_VALID", run_id=run, family=family, seed=92601, data_sha256="1" * 64, refit_scaler_sha="2" * 64,
        selected_epoch=None if family == "RAW" else epoch,
        search_updates=0 if family == "RAW" else 1577,
        search_exposures=0 if family == "RAW" else 25215,
        refit_updates=0 if family == "RAW" or epoch == 0 else 160,
        refit_exposures=0 if family == "RAW" or epoch == 0 else 2548,
        training_status="NO_TRAINING" if family == "RAW" else "SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED")
    row = dict(run_id=run, family=family, actual_C=32, L=512, H=12, seed=92601,
        latent_k=registered, mse=.25 if family == "B0" else .2, mae=.3,
        n_observed=56000, full_V_origins=148, fit_receipt=fit,
        artifact_sha256="a" * 64, manifest_sha256="b" * 64,
        reference_axis_fingerprints={"target_normalized": "t", "observed": "m", "origins": "o", "channel_ids": "c"},
        band_count_binding_sha256="c" * 64, accuracy_status="PASS_FULL_V_FP64_AND_FRESH_CPU6",
        source_identity="NO_FOUNDATION" if family == "DIRECT_G" else "f" * 64,
        F_only_resident_bytes=None, F_only_resident_scope="NOT_SEPARATELY_RECORDED",
        actual_parameter_numel=None, parameter_numel_scope="NOT_SEPARATELY_RECORDED",
        full_C170_comparison=False, B0_B2_C32_structural_comparison=family in ("B0", "B2"),
        cost_status={}, costs={})
    for batch in (1, 8):
        key = str(batch)
        row["cost_status"][key] = "NOT_COMPLETE" if missing_cost and batch == 8 else "PASS_ACTUAL_PHYSICAL_TRIALS"
        summaries = [dict(run_id=run, phase=phase, batch_size=batch, trials=20,
            median_seconds=.031, p90_seconds=.043) for phase in ("timed", "stage_probe")]
        row["costs"][key] = None if missing_cost and batch == 8 else dict(trial_rows=50,
            summaries=summaries, memory_scope=dict(resident_weights_cpu_MiB=190., foundation_instances_created=int(family != "DIRECT_G")),
            guard_sampled_peak_RSS_bytes=1000, model_load_seconds=1.,
            native_F_counters_per_trial=dict(streams=batch * width, physical_streams=batch * width, physical_calls=int(width > 0) * (2 if family == "RAW" and batch == 8 else 1)),
            F_logical_streams=batch * width * 50, F_physical_streams=batch * width * 50,
            F_calls=50 * int(width > 0) * (2 if family == "RAW" and batch == 8 else 1),
            derived_trial_F_scope="INDEPENDENTLY_SUMMED_FROM_50_PUBLIC_TRIAL_ROWS",
            cold_model_load_in_whole_trial=False, recorded_scope="ACTUAL_CPU2_PHYSICAL_PIPELINE",
            receipt=dict(physical_output_and_inverse_dtype="float64"))
    return row


def fixture_audit():
    return dict(schema="INDEPENDENT_PEMS_B0_B2_AUDIT_v1", status="PASS", error=None,
        audit_source_sha256="d" * 64, source48_preserved=True, input_SHA_before_after_equal=True,
        actual_current_owner_absence_bound=True,
        costs_disjoint_from_all_recorded_fit_validation_parity_and_cost_intervals=True,
        whole_MASTER_complete=False, independent_generalization_claimed=False,
        rows=[fixture_row("B0"), fixture_row("B2")], failures=[], skipped=[], inputs=[],
        reference_rows=None, reference_status="NOT_AVAILABLE_OR_NOT_MATCHED",
        reference_inputs=[], reference_failures=[dict(reason="Synthetic reference unavailable")])


class B0B2ReportTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(builder, "B0/B2 aggregate builder is not implemented")

    def test_default_does_no_io_and_missing_trigger_cannot_execute(self):
        with patch("builtins.open", side_effect=AssertionError("Unexpected I/O")), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(builder.main([]), 0)
        self.assertEqual(json.loads(out.getvalue())["input_reads"], 0)
        self.assertFalse(any(name in sys.modules for name in ("numpy", "torch", "matplotlib")))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): builder.main(["--execute"])

    def test_complete_without_baseline_keeps_zero_refit_and_native_precision(self):
        audit = fixture_audit()
        texts = builder.compose(audit, "priority_pems_subset_20261016_v1/toy_audit.json", "e" * 64)
        rows = list(csv.DictReader(io.StringIO(texts["B0_B2_RESULTS_v1.csv"])))
        self.assertEqual([r["family"] for r in rows], ["B0", "B2"])
        self.assertEqual([r["row_scope"] for r in rows], ["NEW_B0_B2", "NEW_B0_B2"])
        self.assertEqual(rows[0]["selected_refit_epoch"], "0.0")
        self.assertEqual(rows[0]["refit_updates"], "0")
        self.assertEqual(float(rows[1]["MSE"]), .2)
        self.assertIn("-0.04999999999999999", texts["B0_B2_KO_v1.md"])
        self.assertIn("NOT_AVAILABLE_OR_NOT_MATCHED", texts["B0_B2_KO_v1.md"])
        self.assertNotIn("epochs", audit["rows"][0]["fit_receipt"])

    def test_valid_peer_is_reported_when_other_peer_or_cost_is_incomplete(self):
        audit = fixture_audit()
        audit.update(status="PARTIAL_VALID_EVIDENCE_PRESERVED", rows=[fixture_row("B2", missing_cost=True)], failures=[dict(reason="Synthetic B0 failure")])
        texts = builder.compose(audit, "priority_pems_subset_20261016_v1/toy.json", "e" * 64)
        row = list(csv.DictReader(io.StringIO(texts["B0_B2_RESULTS_v1.csv"])))[0]
        self.assertEqual((row["family"], row["B8_status"], row["B8_median_seconds"]), ("B2", "NOT_COMPLETE", ""))
        self.assertIn("Synthetic B0 failure", texts["B0_B2_KO_v1.md"])
        self.assertIn("PARTIAL_VALID_EVIDENCE_PRESERVED", texts["B0_B2_KO_v1.md"])

    def test_global_rejection_and_complete_with_missing_peer_are_rejected(self):
        for change in (dict(status="NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED"), dict(error={"reason": "bad"}),
                dict(actual_current_owner_absence_bound=False), dict(rows=[fixture_row("B0")])):
            with self.assertRaises(ValueError): builder.validate_audit(dict(fixture_audit(), **change))

    def test_equal_axes_cannot_hide_another_subset_or_scaler(self):
        for key in ("data_sha256", "refit_scaler_sha"):
            audit=fixture_audit();audit["rows"][1]["fit_receipt"][key]="3" * 64
            with self.assertRaises(ValueError):builder.validate_audit(audit)

    def test_references_remain_reused_and_require_identical_cohort(self):
        audit = fixture_audit()
        audit.update(reference_rows=[fixture_row("RAW"), fixture_row("DIRECT_G")], reference_status="MATCHED_ACTUAL_RAW_G_C32_REFERENCE",
            reference_inputs=[dict(root="public", relative_path="priority_pems_subset_20261016_v1/toy_reference.json", sha256="f" * 64)], reference_failures=[])
        rows = builder.rows_from_audit(audit)
        self.assertEqual([(r["family"], r["row_scope"], r["accuracy_reused"]) for r in rows],
            [("RAW", "REUSED_REFERENCE", True), ("DIRECT_G", "REUSED_REFERENCE", True), ("B0", "NEW_B0_B2", False), ("B2", "NEW_B0_B2", False)])
        audit["reference_rows"][0]["band_count_binding_sha256"] = "e" * 64
        with self.assertRaises(ValueError): builder.validate_audit(audit)

    def test_isolated_memory_and_numel_stay_unavailable(self):
        rows = builder.rows_from_audit(fixture_audit())
        self.assertIsNone(rows[0]["F_only_resident_bytes"])
        self.assertIsNone(rows[0]["parameter_numel"])
        self.assertEqual(rows[0]["B8_resident_CPU_parameter_buffer_MiB"], 190.)
        bad = fixture_audit(); bad["rows"][0]["F_only_resident_bytes"] = 190
        with self.assertRaises(ValueError): builder.validate_audit(bad)

    def test_selected_activity_and_unavailable_cost_cannot_be_invented(self):
        for change in (dict(search_updates=1576), dict(refit_updates=1)):
            audit = fixture_audit(); audit["rows"][0]["fit_receipt"].update(change)
            with self.assertRaises(ValueError): builder.validate_audit(audit)
        audit = fixture_audit(); audit["rows"][0]["cost_status"]["8"] = "NOT_COMPLETE"
        with self.assertRaises(ValueError): builder.validate_audit(audit)

    def test_existing_report_is_rejected_before_input_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            public = Path(temporary); scope = public / "priority_pems_subset_20261016_v1"; scope.mkdir()
            output = scope / "B0_B2_RESULTS_v1.csv"; output.write_text("preserve")
            with self.assertRaises(FileExistsError): builder.execute(public, public, "missing", "a", "b", "c")
            self.assertEqual(output.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
