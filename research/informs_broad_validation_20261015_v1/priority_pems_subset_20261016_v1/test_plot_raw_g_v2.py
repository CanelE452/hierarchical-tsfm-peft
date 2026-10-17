"""Invented stdlib fixtures only; actual aggregates and renderer are not read."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).with_name("plot_raw_g_v2.py")
spec = importlib.util.spec_from_file_location("plot_raw_g", SOURCE)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class PlotTests(unittest.TestCase):
    def setUp(self):
        self.audit = dict(schema="INDEPENDENT_PEMS_RAW_G_AUDIT_v1", status="PASS", error=None, audit_source_sha256=m.AUDITOR_SHA,
            input_SHA_before_after_equal=True, source48_preserved=True, whole_MASTER_complete=False, independent_generalization_claimed=False,
            full_C170_or_B2_or_Bolt_necessity_proven=False, actual_current_owner_absence_bound=True,
            costs_disjoint_from_all_recorded_fit_validation_parity_and_cost_intervals=True, rows=[], cost_conditions=[], failures=[], skipped=[])
        self.report = dict(schema="PEMS_RAW_G_REPORT_BINDINGS_v1", status="FROM_ACTUAL_PEMS_RAW_G_AUDIT_PASS", builder_source_sha256=m.BUILDER_SHA,
            auditor_source_sha256=m.AUDITOR_SHA, input_SHA_before_after_equal=True, source48_preserved=True, whole_MASTER_complete=False,
            independent_generalization_claimed=False, full_subset_experiments_complete=False, completed_accuracy_rows=2, completed_cost_rows=4)
        self.rows = []
        for index, method in enumerate(m.METHODS):
            row = dict(family=method["family"], run_id=method["run_id"], C="32", L="512", H="12", seed="92601", V_origins="148",
                accuracy_status="PASS_FULL_V_FP64_AND_FRESH_CPU6", observed_labels="56789", MSE=repr(.1234567890123456 + index / 10), MAE=repr(.2345678901234567 + index / 10))
            record = dict(family=method["family"], run_id=method["run_id"], accuracy_status=row["accuracy_status"], n_observed=56789,
                mse=float(row["MSE"]), mae=float(row["MAE"]), cost_status={}, costs={})
            for batch in (1, 8):
                median = .0123456789012345 + index / 10 + batch / 100
                row["B" + str(batch) + "_median_seconds"] = repr(median); row["B" + str(batch) + "_status"] = "PASS_ACTUAL_PHYSICAL_TRIALS"
                record["cost_status"][str(batch)] = "PASS_ACTUAL_PHYSICAL_TRIALS"
                record["costs"][str(batch)] = dict(trial_rows=50, summaries=[dict(phase=phase, trials=20, batch_size=batch, median_seconds=median) for phase in ("timed", "stage_probe")])
                self.audit["cost_conditions"].append(dict(run_id=method["run_id"], batch=batch))
            self.rows.append(row); self.audit["rows"].append(record)

    def prepare(self):
        return m.prepare_data(self.rows, self.audit, self.report)

    def partial(self):
        self.audit["status"] = "PARTIAL_VALID_EVIDENCE_PRESERVED"
        self.audit["failures"] = [dict(scope="invented fixture", reason="Same-batch Raw parity not verified")]
        self.report["status"] = "FROM_ACTUAL_PEMS_RAW_G_AUDIT_" + self.audit["status"]

    def test_default_zero_io_and_no_renderer_import(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("No input")), patch.object(m, "module", side_effect=AssertionError("No helper")), patch.object(m, "render", side_effect=AssertionError("No renderer")), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(m.main([]), 0)
        value = m.json.loads(out.getvalue())
        self.assertEqual((value["input_reads"], value["files_written"], value["actual_render"]), (0, 0, 0))
        self.assertFalse(any(name in sys.modules for name in ("numpy", "torch", "matplotlib")))

    def test_two_exact_rows_original_precision_and_explicit_order(self):
        self.rows.reverse(); original = copy.deepcopy(self.rows)
        data = self.prepare()
        self.assertEqual([item["method"]["family"] for item in data], ["RAW", "DIRECT_G"])
        self.assertEqual(data[0]["values"]["MSE"], .1234567890123456)
        self.assertEqual(data[0]["row"]["observed_labels"], "56789")
        self.assertEqual(self.rows, original)

    def test_partial_failed_raw_batch_remains_none_and_g_cost_survives(self):
        self.partial(); raw = self.rows[0]; record = self.audit["rows"][0]
        raw["B8_status"] = "NOT_COMPLETE"; raw["B8_median_seconds"] = ""
        record["cost_status"]["8"] = "NOT_COMPLETE"; record["costs"].pop("8")
        self.audit["cost_conditions"] = [item for item in self.audit["cost_conditions"] if not (item["run_id"] == raw["run_id"] and item["batch"] == 8)]
        self.report["completed_cost_rows"] = 3
        data = self.prepare()
        self.assertIsNone(data[0]["values"]["B8_median_seconds"])
        self.assertEqual(data[1]["values"]["B8_median_seconds"], float(self.rows[1]["B8_median_seconds"]))
        raw["B8_median_seconds"] = "0"
        with self.assertRaises(ValueError):
            self.prepare()

    def test_missing_peer_is_a_display_slot_not_an_accuracy_row(self):
        self.partial(); self.rows.pop(); self.audit["rows"].pop()
        self.audit["cost_conditions"] = self.audit["cost_conditions"][:2]
        self.report.update(completed_accuracy_rows=1, completed_cost_rows=2)
        data = self.prepare()
        self.assertIsNone(data[1]["row"])
        self.assertTrue(all(value is None for value in data[1]["values"].values()))
        self.assertTrue(all(status == "NOT_COMPLETE_NO_VERIFIED_ACCURACY_ROW" for status in data[1]["statuses"].values()))
        self.assertEqual(len(self.rows), 1)

    def test_canonical_scope_status_source_and_duplicate_changes_rejected(self):
        for target, key, value in ((self.audit, "audit_source_sha256", "0" * 64), (self.report, "builder_source_sha256", "0" * 64),
                (self.audit, "source48_preserved", False), (self.report, "whole_MASTER_complete", True), (self.report, "completed_accuracy_rows", 1),
                (self.audit, "actual_current_owner_absence_bound", False), (self.audit, "full_C170_or_B2_or_Bolt_necessity_proven", True),
                (self.rows[0], "C", "170"), (self.rows[0], "accuracy_status", "NOT_COMPLETE"), (self.rows[0], "run_id", self.rows[1]["run_id"])):
            old = target[key]; target[key] = value
            with self.assertRaises(ValueError):
                self.prepare()
            target[key] = old

    def test_accuracy_cost_and_mask_values_must_equal_audit_exactly(self):
        for key, value in (("MSE", repr(float(self.rows[0]["MSE"]) + 1e-12)), ("MAE", "nan"), ("MSE", "-1"),
                ("observed_labels", "56832"), ("B1_median_seconds", "0"), ("B8_status", "NOT_COMPLETE")):
            old = self.rows[0][key]; self.rows[0][key] = value
            with self.assertRaises(ValueError):
                self.prepare()
            self.rows[0][key] = old

    def test_partial_requires_reason_and_no_empty_or_foreign_report(self):
        self.audit["status"] = "PARTIAL_VALID_EVIDENCE_PRESERVED"; self.report["status"] = "FROM_ACTUAL_PEMS_RAW_G_AUDIT_" + self.audit["status"]
        with self.assertRaises(ValueError):
            self.prepare()
        self.audit["failures"] = [dict(reason="Invented fixture")]
        with self.assertRaises(ValueError):
            m.prepare_data([], self.audit, self.report)

    def test_four_panel_artist_binding_exact_values_and_missing_status(self):
        data = self.prepare(); artists = self.artists(data)
        m.validate_artists(artists, data)
        self.assertEqual(sum(len(artist["values"]) for artist in artists), 8)
        for key, value in (("values", [0., 0.]), ("row_statuses", ["NOT_COMPLETE", "NOT_COMPLETE"]), ("missing_run_ids", [m.METHODS[0]["run_id"]])):
            changed = copy.deepcopy(artists); changed[0][key] = value
            with self.assertRaises(ValueError):
                m.validate_artists(changed, data)

    def artists(self, data):
        result = []
        for column in m.COLUMNS:
            present = [(index, item) for index, item in enumerate(data) if item["values"][column] is not None]
            result.append(dict(column=column, positions=[index for index, _ in present], run_ids=[item["method"]["run_id"] for _, item in present],
                values=[item["values"][column] for _, item in present], missing_run_ids=[item["method"]["run_id"] for item in data if item["values"][column] is None],
                row_statuses=[item["statuses"][column] for item in data]))
        return result

    def test_fake_axes_draws_only_measured_bars_and_keeps_zero_real_value(self):
        class Bar:
            def __init__(self, height): self.height = height
            def get_height(self): return self.height
            def set_hatch(self, value): self.hatch = value
        class Axis:
            def bar(self, positions, values, **kwargs):
                self.positions = list(positions); self.values = list(values)
                return [Bar(value) for value in values]
            def set_ylim(self, bottom, top): self.bottom = bottom
            def __getattr__(self, name): return lambda *args, **kwargs: None
        data = self.prepare(); data[0]["values"]["MSE"] = 0.
        data[0]["values"]["B8_median_seconds"] = None; data[0]["statuses"]["B8_median_seconds"] = "NOT_COMPLETE"
        axes = [Axis() for _ in range(4)]; artists = m.draw_panels(axes, data)
        self.assertEqual(axes[0].values[0], 0.)
        self.assertEqual(axes[3].positions, [1])
        self.assertEqual(sum(len(artist["values"]) for artist in artists), 7)
        self.assertTrue(all(axis.bottom == 0 for axis in axes))
        with self.assertRaises(ValueError): m.draw_panels(axes[:3], data)

    def test_canonical_csv_roundtrip_keeps_original_float_and_blank(self):
        class Builder:
            FIELDS = ("value", "missing", "scope")
        blob = m.csv_bytes(Builder, [dict(value=.1234567890123456, missing=None, scope=dict(a=1))])
        self.assertIn(b"0.1234567890123456,,", blob)
        self.assertEqual(len(list(m.csv.DictReader(io.StringIO(blob.decode())))), 1)

    def test_existing_any_output_refused_before_inputs_or_renderer(self):
        with patch.object(Path, "exists", return_value=True), patch.object(m, "module", side_effect=AssertionError("No helper")), patch.object(m, "render", side_effect=AssertionError("No renderer")):
            with self.assertRaises(FileExistsError):
                m.execute("public", "cache", "plot", "report", "audit", "audit-sha", "csv-sha")


if __name__ == "__main__":
    unittest.main()
