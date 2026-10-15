"""Synthetic typed metadata/CSV arithmetic and report-scope regression fixtures."""
from pathlib import Path
import ast
import copy
import csv
import json
import statistics
import tempfile
import unittest
from unittest.mock import patch

import audit_raw_subset_report_v1 as audit
import make_raw_subset_report_v1 as fixture_writer
from test_make_raw_subset_report_v1 import fixture


def values_fixture(folder):
    paths = fixture(Path(folder)); values = {name: audit.read(path) for name, path in paths.items()}
    science = values["science"]
    values["prior_partial"] = audit.read(science["prior_partial"]["path"])
    values["Gram_failure_diagnostic"] = audit.read(science["Gram_failure_diagnostic"]["path"])
    values["preserved_audit_failures"] = []
    for name in audit.FAILURE_SHAS:
        path = Path(folder) / (name + ".json"); record = {"path": str(path), "sha256": audit.sha(path)}; failed = audit.read(path)
        repair = "MD addendum verified as exact path and SHA instead of JSON parsing; no numeric or timer change" if name == "RAW_SUBSET_COST_AUDIT_v1" else "producer-complete bool and complete-C mask used different variable names in fresh auditor; no formula/tolerance/model change"
        values["preserved_audit_failures"].append({"name": name, **record, **failed, "accepted_scientific_PASS": False, "new_forecast_or_training_executed_for_audit": False, "repair_scope": repair})
    for data in values["cost"]["cost"].values():
        for row in data["rows"]:
            for sample in row["samples"]:
                sample["perf_start_ns"] = sample.pop("timer_start_ns"); sample["perf_end_ns"] = sample.pop("timer_end_ns")
                sample["seconds"] = (sample["perf_end_ns"] - sample["perf_start_ns"]) / 1e9
            row["seconds"] = [sample["seconds"] for sample in row["samples"]]
            row["median_seconds"] = statistics.median(row["seconds"]); row["median_seconds_per_window"] = row["median_seconds"] / row["batch"]
        for row in data["rows"]:
            raw = next(other for other in data["rows"] if other["route"] == "RAW_BOLT" and other["batch"] == row["batch"])
            row["paired_RAW_ratios"] = [value / base for value, base in zip(row["seconds"], raw["seconds"])]
            row["paired_RAW_median_ratio"] = statistics.median(row["paired_RAW_ratios"])
    return values


def terminal_fixture(folder, source_values):
    values = copy.deepcopy(source_values); science, verified, cost, cost_audit = (values[name] for name in audit.INPUTS)
    def save(name, value):
        path = Path(folder) / name; path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
        return {"path": str(path), "sha256": audit.sha(path)}
    registry = audit.read(science["selection_registry"]["path"]); registry["failed_paths_at_seal"] = []
    science["selection_registry"] = save("registry_full.json", registry); cost["selection_registry"] = science["selection_registry"]
    for name, basename in (("audit", "audit_raw_subset_comparison_v3.py"), ("cost_audit", "audit_raw_subset_cost_v2.py")):
        path = Path(folder) / basename; path.write_text("# synthetic immutable audit source\n", encoding="utf-8")
        values[name]["audit_source"] = {"path": str(path), "sha256": audit.sha(path)}
    for key in ("source_producer", "source_audit", "plan", "selection_registry"): verified[key] = science[key]
    for item in (cost, cost_audit):
        for key in ("source_producer", "source_audit", "prior_partial", "Gram_failure_diagnostic", "numeric_repair_addendum"): item[key] = science[key]
    science_binding = save("science_complete.json", science); verified["diagnostic"] = science_binding
    audit_binding = save("science_PASS.json", verified)
    for item in (cost, cost_audit): item["raw_subset_diagnostic"] = science_binding; item["raw_subset_audit"] = audit_binding
    cost_binding = save("cost_complete.json", cost); cost_audit["cost"] = cost_binding
    checked_cost_binding = save("cost_PASS.json", cost_audit)
    bindings = dict(science=science_binding, audit=audit_binding, cost=cost_binding, cost_audit=checked_cost_binding, audit_source=verified["audit_source"], cost_audit_source=cost_audit["audit_source"])
    bindings.update({key: science[key] for key in ("prior_partial", "Gram_failure_diagnostic", "numeric_repair_addendum")})
    failure_shas = {}
    for row in values["preserved_audit_failures"]:
        bindings[row["name"]] = {key: row[key] for key in ("path", "sha256")}; failure_shas[row["name"]] = row["sha256"]
    return {"inputs": bindings}, {name: bindings[name]["sha256"] for name in audit.INPUTS}, failure_shas


class RawSubsetReportVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name); self.values = values_fixture(self.folder)
        self.text_output = audit.HERE / "reports" / "synthetic_verifier_not_created"

    def test_all_nineteen_tables_independently_reconstructed_from_synthetic_JSON(self):
        checked = audit.independent_tables(self.values); produced = fixture_writer.collect_tables(self.values)
        self.assertEqual(checked, produced)
        self.assertEqual({name: len(rows) for name, rows in checked.items()}, audit.COUNTS)
        self.assertEqual(len(checked["GREEDY_CANDIDATES"]), 83); self.assertEqual(len(checked["GREEDY_RESPONSE_SCORES"]), 1285)
        self.assertEqual(len(checked["AUDIT_FAILURE_PRESERVATION"]), 4)
        json.dumps(checked, allow_nan=False)

    def test_every_typed_csv_cell_nested_raw_metric_and_SHA_matches(self):
        expected = audit.independent_tables(self.values)
        counts = {}
        for name, rows in fixture_writer.collect_tables(self.values).items():
            path = self.folder / (name + ".csv"); fixture_writer.write_csv(path, rows)
            counts[name] = audit.check_csv(path, name, expected[name])
        self.assertGreater(sum(counts.values()), 100000)

    def test_csv_wrong_scalar_nested_observed_count_bool_and_source_SHA_rejected(self):
        tables = audit.independent_tables(self.values)
        mutations = (("RELATIVE_TO_RAW", "MSE_ratio_to_RAW", 999.), ("COMPARISON", "observed_counts", [0] * 17), ("NEW_FIXED_D_DIAGNOSTIC", "MAE_is_minimum", True), ("GLOBAL_DECODER_STATES", "state", {"path": "wrong.npz", "sha256": "wrong"}))
        for index, (name, field, value) in enumerate(mutations):
            rows = copy.deepcopy(tables[name]); rows[0][field] = value; path = self.folder / (str(index) + ".csv"); fixture_writer.write_csv(path, rows)
            with self.subTest(name=name), self.assertRaises(ValueError): audit.check_csv(path, name, tables[name])

    def test_duplicate_or_missing_checkpoint_row_rejected(self):
        rows = audit.independent_tables(self.values)["CHECKPOINTS"]
        for case in (rows[:-1], [*rows[:-1], rows[0]]):
            path = self.folder / (str(len(case)) + "_" + str(id(case)) + ".csv"); fixture_writer.write_csv(path, case)
            with self.assertRaises(ValueError): audit.check_csv(path, "CHECKPOINTS", rows)

    def test_timer_ns_and_two_ratio_statistics_are_not_taken_on_producer_authority(self):
        values = copy.deepcopy(self.values); row = values["cost"]["cost"]["robin"]["rows"][1]
        row["samples"][0]["perf_end_ns"] += 1000
        with self.assertRaisesRegex(ValueError, "INDEPENDENT_TIMER_NS"): audit.independent_tables(values)
        values = copy.deepcopy(self.values); row = values["cost"]["cost"]["robin"]["rows"][1]; row["paired_RAW_median_ratio"] += .1
        with self.assertRaisesRegex(ValueError, "INDEPENDENT_TIMER_NS"): audit.independent_tables(values)

    def test_MD_actual_scores_floors_masks_native_CP_and_batch_vs_window_cost(self):
        tables = audit.independent_tables(self.values); text = fixture_writer.make_text(self.values, tables, {}, self.text_output)
        self.assertGreater(audit.check_MD_numeric(text, tables), 40)
        changed = text.replace("B8 ms/window", "B8 batch ms", 1)
        # Numeric mutation targets a displayed actual cost; changed headings are also reviewed through source.
        row = next(row for row in tables["WHOLE_COST"] if row["dataset"] == "robin" and row["route"] == "PCA_BOLT" and row["batch"] == 8)
        changed = changed.replace(f"{row['median_seconds_per_window']*1000:14.7f}", "   999.0000000", 1)
        with self.assertRaisesRegex(ValueError, "MD_NUMERIC"): audit.check_MD_numeric(changed, tables)

    def test_MD_cannot_convert_real_fixed_D_or_exposed_DEV_to_generalization_claim(self):
        tables = audit.independent_tables(self.values); text = fixture_writer.make_text(self.values, tables, {}, self.text_output)
        for phrase in ("MAE·이질적 원단위 MSE를 각 목적의 최소라고 부르지 않는다", "이전 DEV·oracle를 본 뒤 진행한 탐색적 대조", "JSONDecodeError로 FAILED_PRESERVED"):
            with self.subTest(phrase=phrase), self.assertRaises(ValueError): audit.check_MD_numeric(text.replace(phrase, "altered scope", 1), tables)

    def test_actual_source_overwrite_and_metadata_boundaries_without_generation(self):
        source = Path(fixture_writer.__file__).read_text(encoding="utf-8-sig")
        self.assertTrue(audit.source_contract(source)["overwrite_refusal_checked"])
        with self.assertRaises(ValueError): audit.source_contract(source.replace("not output.exists()", "True", 1))

    def test_verifier_source_never_imports_generator_scientific_runtime_or_opens_NPZ(self):
        tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
        imports = [alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names]
        self.assertNotIn("make_raw_subset_report_v1", imports); self.assertNotIn("numpy", imports); self.assertNotIn("torch", imports)
        self.assertNotIn("np.load", Path(audit.__file__).read_text(encoding="utf-8"))

    def test_full_four_JSON_terminal_chain_without_any_scientific_file(self):
        manifest, hashes, failures = terminal_fixture(self.folder, self.values)
        with patch.object(audit, "FAILURE_SHAS", failures): checked = audit.terminal(manifest, hashes)
        self.assertEqual(len(checked["preserved_audit_failures"]), 4)
        self.assertFalse(any(row["accepted_scientific_PASS"] for row in checked["preserved_audit_failures"]))

    def test_terminal_wrong_SHA_or_wrong_new_cost_parent_is_rejected(self):
        manifest, hashes, failures = terminal_fixture(self.folder, self.values)
        with patch.object(audit, "FAILURE_SHAS", failures):
            wrong = dict(hashes); wrong["cost_audit"] = "wrong"
            with self.assertRaisesRegex(ValueError, "REGISTERED_ACTUAL_JSON_SHA"): audit.terminal(manifest, wrong)
            path = Path(manifest["inputs"]["cost_audit"]["path"]); value = audit.read(path); value["raw_subset_diagnostic"]["sha256"] = "wrong"
            path.write_text(json.dumps(value), encoding="utf-8"); hashes["cost_audit"] = manifest["inputs"]["cost_audit"]["sha256"] = audit.sha(path)
            with self.assertRaisesRegex(ValueError, "EXACT_METADATA_BINDING"): audit.terminal(manifest, hashes)

    def artifact_fixture(self):
        from PIL import Image
        folder = self.folder / "synthetic_artifacts"; folder.mkdir(); (folder / "tables").mkdir(); (folder / "figures").mkdir()
        (folder / "REPORT_KO.md").write_text("# synthetic containers only\n", encoding="utf-8")
        for name in audit.COUNTS: (folder / "tables" / (name + ".csv")).write_text("synthetic\n", encoding="utf-8")
        for name in audit.FIGURES:
            Image.new("RGB", (3, 2), "white").save(folder / "figures" / (name + ".png"))
            (folder / "figures" / (name + ".svg")).write_text('<svg xmlns="http://www.w3.org/2000/svg" width="3" height="2"></svg>', encoding="utf-8")
        files = [{"path": path.relative_to(folder).as_posix(), "sha256": audit.sha(path), "bytes": path.stat().st_size} for path in folder.rglob("*") if path.is_file()]
        return folder, {"files": files}

    def test_all_thirtytwo_container_files_and_six_PNG_SVG_pairs(self):
        folder, manifest = self.artifact_fixture()
        self.assertEqual(audit.check_files(folder, manifest), 32)
        parsed = audit.check_figures(folder); self.assertEqual(len(parsed), 6)
        self.assertTrue(all(item["PNG_dimensions"] == [3, 2] and item["SVG_parsed"] for item in parsed))

    def test_manifest_escape_duplicate_SHA_or_extra_file_rejected(self):
        folder, manifest = self.artifact_fixture()
        for case in ("escape", "duplicate", "SHA"):
            wrong = copy.deepcopy(manifest)
            if case == "escape": wrong["files"][0]["path"] = "../outside.md"
            elif case == "duplicate": wrong["files"].append(wrong["files"][0])
            else: wrong["files"][0]["sha256"] = "wrong"
            with self.subTest(case=case), self.assertRaises(ValueError): audit.check_files(folder, wrong)
        (folder / "unregistered.txt").write_text("extra", encoding="utf-8")
        with self.assertRaises(ValueError): audit.check_files(folder, manifest)

    def test_invalid_PNG_or_SVG_payloads_are_parsed_not_accepted_from_extension(self):
        folder, _ = self.artifact_fixture()
        svg = folder / "figures" / (audit.FIGURES[0] + ".svg"); svg.write_text("not XML", encoding="utf-8")
        with self.assertRaises(Exception): audit.check_figures(folder)
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
        png = folder / "figures" / (audit.FIGURES[0] + ".png"); png.write_text("not PNG", encoding="utf-8")
        with self.assertRaises(Exception): audit.check_figures(folder)

    def test_relative_links_resolve_spaces_and_missing_files_fail(self):
        path = self.folder / "sample image.svg"; path.write_text("synthetic", encoding="utf-8")
        report = self.folder / "links.md"; report.write_text("![sample](sample%20image.svg)\n", encoding="utf-8")
        found, embedded = audit.relative_links(report); self.assertEqual(len(found), 1); self.assertEqual(embedded, [path.resolve()])
        report.write_text("[missing](missing.md)", encoding="utf-8")
        with self.assertRaises(ValueError): audit.relative_links(report)


if __name__ == "__main__": unittest.main()
