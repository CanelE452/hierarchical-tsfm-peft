"""Invented report metadata only; actual N6 audit/data remain unopened."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name("build_raw_g_report_v2.py")
spec = importlib.util.spec_from_file_location("raw_g_report_test", SOURCE)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def audit_fixture():
    rows = []
    for family in ("RAW", "DIRECT_G"):
        fit = dict(selected_epoch=None if family == "RAW" else 0., search_updates=0 if family == "RAW" else 1577,
            search_exposures=0 if family == "RAW" else 25215, refit_updates=0, refit_exposures=0,
            training_status="NO_TRAINING" if family == "RAW" else "SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED")
        costs = {}
        for batch in (1, 8):
            streams = batch * 32 if family == "RAW" else 0; calls = (1 if batch == 1 else 2) if family == "RAW" else 0
            costs[str(batch)] = dict(trial_rows=50, derived_trial_F_scope="INDEPENDENTLY_SUMMED_FROM_50_PUBLIC_TRIAL_ROWS", cold_model_load_in_whole_trial=False,
                summaries=[dict(phase=phase, trials=20, median_seconds=.001, p90_seconds=.002) for phase in ("timed", "stage_probe")],
                memory_scope=dict(resident_weights_cpu_MiB=1., foundation_instances_created=1 if family == "RAW" else 0), guard_sampled_peak_RSS_bytes=1234,
                model_load_seconds=.3, native_F_counters_per_trial=dict(streams=streams, physical_streams=streams, physical_calls=calls),
                F_logical_streams=50 * streams, F_physical_streams=50 * streams, F_calls=50 * calls, recorded_scope="WARM_WHOLE_PHYSICAL_ROUTE_WITH_SEPARATE_INTRUSIVE_STAGE_PROBE")
        rows.append(dict(family=family, run_id=m.RUNS[family], actual_C=32, L=512, H=12, seed=92601, full_V_origins=148, n_observed=12345,
            accuracy_status="PASS_FULL_V_FP64_AND_FRESH_CPU6", full_C170_or_B2_comparison=False, independent_generalization_claimed=False,
            mse=.12345678901234566, mae=.2, actual_parameter_numel=None, parameter_numel_scope="NOT_RECORDED_BY_NATIVE_ROUTE_MEMORY_API",
            F_only_resident_bytes=None, F_only_resident_scope="NOT_SEPARATELY_RECORDED", fit_receipt=fit, artifact_sha256="a" * 64, manifest_sha256="b" * 64,
            source_identity="c" * 64 if family == "RAW" else "NO_FOUNDATION", cost_status={"1": "PASS_ACTUAL_PHYSICAL_TRIALS", "8": "PASS_ACTUAL_PHYSICAL_TRIALS"}, costs=costs))
    return dict(schema="INDEPENDENT_PEMS_RAW_G_AUDIT_v1", status="PASS", error=None, source48_preserved=True, input_SHA_before_after_equal=True,
        whole_MASTER_complete=False, independent_generalization_claimed=False, full_C170_or_B2_or_Bolt_necessity_proven=False,
        actual_current_owner_absence_bound=True, costs_disjoint_from_all_recorded_fit_validation_parity_and_cost_intervals=True,
        rows=rows, failures=[], skipped=[], cost_conditions=[cost for row in rows for cost in row["costs"].values()])


class Tests(unittest.TestCase):
    def test_default_zero_io(self):
        out = io.StringIO()
        with patch("builtins.open", side_effect=AssertionError("No input")), patch.object(m, "sha", side_effect=AssertionError("No source read")), contextlib.redirect_stdout(out):
            self.assertEqual(m.main([]), 0)
        self.assertEqual(m.json.loads(out.getvalue())["input_reads"], 0)

    def test_actual_precision_and_separate_memory_scopes(self):
        audit = audit_fixture(); rows = m.rows_from_audit(audit)
        self.assertEqual(rows[0]["MSE"], .12345678901234566)
        self.assertIsNone(rows[0]["F_only_resident_bytes"])
        self.assertEqual(rows[0]["B8_native_F_counters_per_trial"]["physical_calls"], 2)
        self.assertEqual(rows[1]["B8_native_F_counters_per_trial"]["physical_calls"], 0)
        self.assertEqual(rows[1]["F_identity"], "NO_FOUNDATION")
        self.assertFalse(rows[0]["B8_cold_load_in_whole_trial"])

    def test_raw_cost_failure_preserves_g_and_valid_raw_accuracy(self):
        audit = audit_fixture(); audit["status"] = "PARTIAL_VALID_EVIDENCE_PRESERVED"
        audit["rows"][0]["cost_status"]["8"] = "NOT_COMPLETE"; del audit["rows"][0]["costs"]["8"]
        audit["failures"] = [dict(run_id=m.RUNS["RAW"], stage="COST_B8", status="NOT_COMPLETE", reason="Invented parity failure")]
        rows = m.rows_from_audit(audit)
        self.assertEqual(len(rows), 2); self.assertNotIn("B8_median_seconds", rows[0]); self.assertIn("B8_median_seconds", rows[1])
        text = m.markdown(audit, rows, m.NAME + "/INVENTED_AUDIT.json", "a" * 64)
        self.assertIn("PARTIAL_VALID_EVIDENCE_PRESERVED", text); self.assertIn("UNKNOWN", text); self.assertIn("INVENTED_AUDIT.json", text)
        self.assertIn(".12345678901234566", text); self.assertNotIn("](/", text)
        bad = copy.deepcopy(audit); bad["status"] = "PASS"
        with self.assertRaises(ValueError): m.rows_from_audit(bad)

    def test_reject_unproven_partial_or_master_and_synthetic_zero_cost(self):
        for mutation in ({"status": "NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED"}, {"source48_preserved": False}, {"whole_MASTER_complete": True}, {"full_C170_or_B2_or_Bolt_necessity_proven": True}):
            audit = audit_fixture(); audit.update(mutation)
            with self.assertRaises(ValueError): m.rows_from_audit(audit)
        audit = audit_fixture(); audit["status"] = "PARTIAL_VALID_EVIDENCE_PRESERVED"
        with self.assertRaises(ValueError): m.rows_from_audit(audit)
        audit = audit_fixture(); audit["rows"][0]["cost_status"]["8"] = "NOT_COMPLETE"
        with self.assertRaises(ValueError): m.rows_from_audit(audit)

    def test_g_zero_selected_refit_is_not_missing_training(self):
        audit = audit_fixture(); rows = m.rows_from_audit(audit)
        self.assertEqual(rows[1]["selected_refit_epoch"], 0.)
        self.assertEqual(rows[1]["search_updates"], 1577)
        bad = copy.deepcopy(audit); bad["rows"][1]["fit_receipt"]["search_updates"] = 1576
        with self.assertRaises(ValueError): m.rows_from_audit(bad)

    def test_fake_create_new_outputs_have_exact_sha_and_working_relative_link(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); public = base / "public"; cache = base / "cache"
            scope = public / m.NAME; scope.mkdir(parents=True); cache.mkdir()
            helper = public / m.INPUT_HELPER; helper.parent.mkdir(parents=True); helper.write_bytes(b"invented helper source")
            builder = scope / "build_raw_g_report_v2.py"; builder.write_bytes(SOURCE.read_bytes())
            auditor = scope / "audit_raw_g_v2.py"; auditor.write_bytes(b"invented auditor source")
            audit = audit_fixture(); audit.update(audit_source_sha256=m.sha(auditor), inputs=[])
            audit_path = scope / "INVENTED_AUDIT.json"; audit_path.write_text(m.json.dumps(audit), encoding="utf-8")
            class Inputs:
                def __init__(self, public, cache): self.roots = dict(public=Path(public), cache=Path(cache)); self.files = {}
                def path(self, root, name): return self.roots[root] / name
                def capture(self, root, name, expected):
                    path = self.path(root, name); actual = m.sha(path)
                    if actual != expected: raise ValueError("Invented input SHA mismatch")
                    self.files[(root, name)] = actual; return path
                def json(self, root, name, expected): return m.json.loads(self.capture(root, name, expected).read_text(encoding="utf-8"))
                def preserved(self): return all(m.sha(self.path(root, name)) == sha for (root, name), sha in self.files.items())
            fake_module = types.SimpleNamespace(Inputs=Inputs)
            fake_spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda module: None))
            with patch.object(m, "INPUT_HELPER_SHA", m.sha(helper)), patch.object(m.importlib.util, "spec_from_file_location", return_value=fake_spec), patch.object(m.importlib.util, "module_from_spec", return_value=fake_module):
                binding = m.execute(public, cache, m.NAME + "/INVENTED_AUDIT.json", m.sha(audit_path), m.sha(auditor), m.sha(SOURCE))
                for output in binding["outputs"]:
                    path = public / output["relative_path"]
                    self.assertEqual(m.sha(path), output["sha256"]); self.assertEqual(path.stat().st_size, output["bytes"])
                self.assertIn("[독립 감사](INVENTED_AUDIT.json)", (scope / m.OUTPUTS[1]).read_text(encoding="utf-8"))
                before = {name: m.sha(scope / name) for name in m.OUTPUTS}
                with self.assertRaises(FileExistsError): m.execute(public, cache, m.NAME + "/INVENTED_AUDIT.json", m.sha(audit_path), m.sha(auditor), m.sha(SOURCE))
                self.assertEqual(before, {name: m.sha(scope / name) for name in m.OUTPUTS})


if __name__ == "__main__":
    unittest.main()
