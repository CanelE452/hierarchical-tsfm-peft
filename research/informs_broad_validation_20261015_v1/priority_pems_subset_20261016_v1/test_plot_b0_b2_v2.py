"""Tiny stdlib-only plot metadata/artist fixtures, never a renderer test."""
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


def load(name, filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


plot = load("prepared_b0_b2_plot", "plot_b0_b2_v2.py") if Path(__file__).with_name("plot_b0_b2_v2.py").exists() else None
fixtures = load("b0_b2_plot_synthetic_fixtures", "test_build_b0_b2_report.py")
builder = fixtures.builder


def fixture(references=True, partial=False):
    audit = fixtures.fixture_audit(); audit["audit_source_sha256"] = plot.AUDITOR_SHA
    if references:
        audit.update(reference_rows=[fixtures.fixture_row("RAW"), fixtures.fixture_row("DIRECT_G")],
            reference_status="MATCHED_ACTUAL_RAW_G_C32_REFERENCE", reference_failures=[],
            reference_inputs=[dict(root="public", relative_path=plot.NAME + "/synthetic_reference.json", sha256="f" * 64)])
    if partial:
        audit.update(status="PARTIAL_VALID_EVIDENCE_PRESERVED", rows=[fixtures.fixture_row("B2", missing_cost=True)],
            failures=[dict(reason="Synthetic B0 incomplete")])
    texts = builder.compose(audit, plot.NAME + "/synthetic_audit.json", "a" * 64)
    rows = list(csv.DictReader(io.StringIO(texts[builder.OUTPUTS[0]])))
    report = dict(schema="PEMS_B0_B2_REPORT_BINDINGS_v1", status="FROM_ACTUAL_PEMS_B0_B2_AUDIT_" + audit["status"],
        auditor_source_sha256=plot.AUDITOR_SHA, builder_source_sha256=plot.BUILDER_SHA,
        input_SHA_before_after_equal=True, source48_preserved=True, whole_MASTER_complete=False,
        independent_generalization_claimed=False, full_subset_experiments_complete=False,
        completed_new_accuracy_rows=len(audit["rows"]), reused_reference_rows=len(audit.get("reference_rows") or []),
        completed_new_cost_rows=sum(value == "PASS_ACTUAL_PHYSICAL_TRIALS" for row in audit["rows"] for value in row["cost_status"].values()),
        reference_status=audit["reference_status"])
    return rows, audit, report


class Bar:
    def __init__(self, value): self.value = value
    def get_height(self): return self.value
    def set_hatch(self, value): pass


class Axis:
    def __init__(self): self.bars = []; self.ticks = None
    def bar(self, positions, values, **kwargs):
        self.bars.append((positions, values)); return [Bar(value) for value in values]
    def set_xticks(self, positions, labels): self.ticks = (list(positions), labels)
    def __getattr__(self, name): return lambda *args, **kwargs: None


class B0B2PlotTests(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(plot, "B0/B2 plot source not implemented")

    def test_default_zero_io_and_trigger_required(self):
        with patch("builtins.open", side_effect=AssertionError("I/O")), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(plot.main([]), 0)
        self.assertEqual(json.loads(out.getvalue())["input_reads"], 0)
        self.assertFalse(any(name in sys.modules for name in ("numpy", "torch", "matplotlib")))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): plot.main(["--execute"])

    def test_complete_four_methods_and_16_exact_artist_values(self):
        data = plot.prepare_data(*fixture()); axes = [Axis() for _ in range(4)]
        artists = plot.draw_panels(axes, data)
        self.assertEqual([item["method"]["family"] for item in data], ["RAW", "DIRECT_G", "B0", "B2"])
        self.assertEqual(sum(len(item["values"]) for item in artists), 16)
        self.assertTrue(all(axis.ticks[0] == [0, 1, 2, 3] for axis in axes))
        self.assertEqual(artists[0]["values"], [.2, .2, .25, .2])

    def test_partial_has_no_fake_baselines_or_zero_cost(self):
        data = plot.prepare_data(*fixture(references=False, partial=True))
        self.assertEqual([item["row"] is not None for item in data], [False, False, False, True])
        self.assertEqual(data[3]["values"], dict(MSE=.2, MAE=.3, B1_median_seconds=.031, B8_median_seconds=None))
        artists = plot.draw_panels([Axis() for _ in range(4)], data)
        self.assertEqual(artists[3]["values"], [])
        self.assertEqual(artists[3]["missing_run_ids"], [item["method"]["run_id"] for item in data])

    def test_zero_refit_is_valid_and_reused_rows_remain_reused(self):
        rows, audit, report = fixture(); data = plot.prepare_data(rows, audit, report)
        self.assertEqual(data[2]["row"]["selected_refit_epoch"], "0.0")
        self.assertEqual(data[0]["row"]["accuracy_reused"], "True")
        rows[0]["row_scope"] = "NEW_B0_B2"
        with self.assertRaises(ValueError): plot.prepare_data(rows, audit, report)

    def test_producer_and_partial_status_fail_closed(self):
        for owner, field, value in (("audit", "audit_source_sha256", "0" * 64), ("report", "builder_source_sha256", "0" * 64),
                ("audit", "status", "NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED"), ("report", "completed_new_accuracy_rows", 1)):
            rows, audit, report = fixture(); (audit if owner == "audit" else report)[field] = value
            with self.assertRaises(ValueError): plot.prepare_data(rows, audit, report)

    def test_foreign_duplicate_or_missing_csv_peer_rejected(self):
        for mutate in (lambda rows: rows.append(copy.deepcopy(rows[0])), lambda rows: rows.pop(), lambda rows: rows[0].update(run_id="foreign")):
            rows, audit, report = fixture(); mutate(rows)
            with self.assertRaises(ValueError): plot.prepare_data(rows, audit, report)

    def test_csv_must_preserve_accuracy_width_and_actual_count(self):
        for field, value in (("MSE", ".20000001"), ("F_width", "32"), ("observed_labels", "56832"), ("device", "cuda")):
            rows, audit, report = fixture(); rows[2][field] = value
            with self.assertRaises(ValueError): plot.prepare_data(rows, audit, report)

    def test_missing_cost_cannot_be_zero_or_nonempty(self):
        for value in ("0", "0.01", "NA"):
            rows, audit, report = fixture(partial=True); rows[-1]["B8_median_seconds"] = value
            with self.assertRaises(ValueError): plot.prepare_data(rows, audit, report)

    def test_ordinary20_median_exact_and_stage_not_substituted(self):
        rows, audit, report = fixture(); rows[-1]["B8_median_seconds"] = ".032"
        with self.assertRaises(ValueError): plot.prepare_data(rows, audit, report)
        rows, audit, report = fixture(); audit["rows"][-1]["costs"]["8"]["summaries"][0]["phase"] = "stage_probe"
        with self.assertRaises(ValueError): plot.prepare_data(rows, audit, report)

    def test_artist_value_and_absence_mutation_rejected(self):
        data = plot.prepare_data(*fixture(partial=True)); artists = plot.draw_panels([Axis() for _ in range(4)], data)
        bad = copy.deepcopy(artists); bad[0]["values"][0] += .1
        with self.assertRaises(ValueError): plot.validate_artists(bad, data)
        bad = copy.deepcopy(artists); bad[3]["missing_run_ids"] = []
        with self.assertRaises(ValueError): plot.validate_artists(bad, data)

    def test_existing_output_rejected_before_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            public = Path(temporary); scope = public / plot.NAME; scope.mkdir()
            path = scope / plot.VIS_MD; path.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError): plot.execute(public, public, "x", "x", "x", "x", "x")
            self.assertEqual(path.read_bytes(), b"preserve")


if __name__ == "__main__": unittest.main()
