"""Synthetic producer report versus independent verifier, with primitive tampering."""
from pathlib import Path
import ast
import builtins
import copy
import csv
import json
import statistics
import tempfile
import unittest
from unittest.mock import patch

import audit_causal_lag_report_v1 as audit
import make_causal_lag_report_v1 as synthetic_producer
from test_make_causal_lag_report_v1 import fixture_values, write


class CausalLagReportAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="independent_causal_report_", dir=audit.HERE); cls.addClassCleanup(cls.temp.cleanup)
        cls.folder = Path(cls.temp.name); paths, config = fixture_values(cls.folder)
        cost = json.loads(paths["cost"].read_text(encoding="utf-8"))
        for data in cost["cost"].values():
            for row in data["rows"]:
                row["seconds"] = [(s["perf_end_ns"] - s["perf_start_ns"]) / 1e9 for s in row["samples"]]
                for value, sample in zip(row["seconds"], row["samples"]): sample["seconds"] = value
            for row in data["rows"]:
                raw = next(r for r in data["rows"] if r["route"] == "RAW_BOLT" and r["batch"] == row["batch"])
                row["paired_RAW_ratios"] = [a / b for a, b in zip(row["seconds"], raw["seconds"])]
                row["paired_RAW_median_ratio"] = statistics.median(row["paired_RAW_ratios"])
                row["median_seconds"] = statistics.median(row["seconds"]); row["median_seconds_per_window"] = row["median_seconds"] / row["batch"]
        cost_binding = write(paths["cost"], cost)
        checked = json.loads(paths["cost_audit"].read_text(encoding="utf-8")); checked["cost"] = cost_binding; write(paths["cost_audit"], checked)
        config["FIXED_SHA"] = {name: audit.sha(path) for name, path in paths.items()}
        cls.output = cls.folder / "reports/toy"
        with patch.multiple(synthetic_producer, **config): cls.manifest = synthetic_producer.generate(paths, cls.output)
        cls.registration = {"generator": audit.sha(synthetic_producer.__file__), "manifest": audit.sha(cls.output / "MANIFEST.json"), "report": audit.sha(cls.output / "REPORT_KO.md"), "inputs": config["FIXED_SHA"], "auditors": config["AUDITOR_SOURCES"], "failure": config["PRESERVED_FAILURES"]["science_audit_v1"]["sha256"], "repair": {key: value["sha256"] for key, value in config["REPAIR_EVIDENCE"].items()}}
        cls.values = audit.terminal(cls.manifest, cls.registration)
        cls.tables = audit.independent_tables(cls.values)

    def test_full_synthetic_producer_JSON_to_independent_all_CSV_MD_links_and_images(self):
        original = builtins.open
        def safe_open(path, *args, **kwargs):
            if str(path).lower().endswith((".npz", ".pt", ".pth", ".safetensors")): raise AssertionError("scientific asset opened")
            return original(path, *args, **kwargs)
        with patch("builtins.open", side_effect=safe_open): result = audit.verify(self.output, self.registration)
        self.assertEqual(result["status"], "PASS"); self.assertEqual(result["files_checked"], 31)
        self.assertEqual(result["PNG_count"], 6); self.assertEqual(result["SVG_count"], 6)
        self.assertEqual(result["CSV_rows"]["COMPARISON"], 336); self.assertEqual(result["CSV_rows"]["WHOLE_COST_SAMPLES"], 676)
        self.assertFalse(result["report_generator_imported"]); self.assertFalse(result["scientific_NPZ_or_model_opened"])
        json.dumps(result, allow_nan=False)

    def test_pending_gate_does_not_open_report_or_assets(self):
        config = dict(self.registration, manifest=None)
        with patch("builtins.open", side_effect=AssertionError("closed gate read")), self.assertRaisesRegex(ValueError, "NOT_YET_REGISTERED"): audit.verify(Path("missing"), config)

    def test_nested_original_observed_count_or_source_SHA_tamper_rejected(self):
        for field in ("observed_counts", "nested_sha"):
            name = "COMPARISON" if field == "observed_counts" else "F_CACHE_LINEAGE"
            source = self.output / ("tables/" + name + ".csv")
            with source.open(encoding="utf-8-sig", newline="") as handle: reader = csv.DictReader(handle); rows, columns = list(reader), reader.fieldnames
            if field == "observed_counts":
                value = json.loads(rows[0]["observed_counts"]); value[0] += 1; rows[0]["observed_counts"] = json.dumps(value)
            else:
                value = json.loads(rows[0]["input_source"]); value["sha256"] = "wrong"; rows[0]["input_source"] = json.dumps(value)
            target = self.folder / (field + ".csv")
            with target.open("w", encoding="utf-8-sig", newline="") as handle: writer = csv.DictWriter(handle, columns); writer.writeheader(); writer.writerows(rows)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "CSV_SOURCE_VALUE_MISMATCH"): audit.check_csv(target, name, self.tables[name])

    def test_omitted_row_or_extra_column_is_not_accepted(self):
        source = self.output / "tables/REGISTERED_ADOPTION.csv"
        with source.open(encoding="utf-8-sig", newline="") as handle: reader = csv.DictReader(handle); rows, columns = list(reader), reader.fieldnames
        target = self.folder / "omitted.csv"
        with target.open("w", encoding="utf-8-sig", newline="") as handle: writer = csv.DictWriter(handle, columns); writer.writeheader(); writer.writerows(rows[:-1])
        with self.assertRaisesRegex(ValueError, "ROWS"): audit.check_csv(target, "REGISTERED_ADOPTION", self.tables["REGISTERED_ADOPTION"])

    def test_MD_numeric_full_mask_and_repair_scope_tamper(self):
        text = (self.output / "REPORT_KO.md").read_text(encoding="utf-8")
        for before, after in (("Raw보다 full-observed", "Raw보다 complete-C"), ("정확한 zero 열만 LS 계산에서 제외", "작은 열은 전부 삭제"), ("독립 검산28,365/실패0", "최초 예측 목적 D")):
            self.assertIn(before, text)
            with self.subTest(before=before), self.assertRaises(ValueError): audit.check_MD(text.replace(before, after), self.tables)

    def test_two_cost_ratios_are_independent_and_ns_changed_rejected(self):
        row = next(r for r in self.tables["WHOLE_COST"] if r["route"] != "RAW_BOLT")
        self.assertNotEqual(row["ratio_of_RAW_medians"], row["median_paired_RAW_ratio"])
        values = copy.deepcopy(self.values); values["cost"]["cost"]["robin"]["rows"][1]["samples"][0]["perf_end_ns"] += 1
        with self.assertRaisesRegex(ValueError, "CSV_SOURCE_VALUE_MISMATCH"): audit.independent_tables(values)

    def test_original_false_bound_flags_not_changed_into_PASS(self):
        values = copy.deepcopy(self.values); method = values["science"]["datasets"]["robin"]["spaces"]["PCA"]["conditions"]["L7_RECON"]["splits"]["val"]["diagnostics"]["methods"]["ACTUAL_F"]
        method["real_trace_bound_inequality_pass"] = False; method["FP32_bound_with_measured_allowance_pass"] = False
        rows = audit.independent_tables(values)["RETAINED_DISCARDED"]
        row = next(r for r in rows if r["dataset"] == "robin" and r["space"] == "PCA" and r["condition"] == "L7_RECON" and r["split"] == "val" and r["method"] == "ACTUAL_F")
        self.assertFalse(row["real_trace_bound_inequality_pass"]); self.assertTrue(row["retained_output_trace_bound_inequality_pass"])

    def test_generator_source_contract_refuses_overwrite_and_cached_scientific_calls(self):
        text = Path(synthetic_producer.__file__).read_text(encoding="utf-8")
        self.assertTrue(audit.source_contract(text)["overwrite_refusal"])
        with self.assertRaises(ValueError): audit.source_contract(text.replace("not output.exists()", "True"))
        with self.assertRaises(ValueError): audit.source_contract("import torch\n" + text)
        calls = {ast.unparse(node.func) for node in ast.walk(ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))) if isinstance(node, ast.Call)}
        self.assertFalse({"collect_tables", "generate", "np.load", "torch.load"} & calls)

    def test_manifest_hash_or_file_sha_tampering_rejected(self):
        registration = dict(self.registration, manifest="0" * 64)
        with self.assertRaisesRegex(ValueError, "REPORT_AND_MANIFEST_SHA"): audit.verify(self.output, registration)
        manifest = copy.deepcopy(self.manifest); manifest["files"][0]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "FILE_SHA_BYTES"): audit.check_files(self.output, manifest)


if __name__ == "__main__": unittest.main()
