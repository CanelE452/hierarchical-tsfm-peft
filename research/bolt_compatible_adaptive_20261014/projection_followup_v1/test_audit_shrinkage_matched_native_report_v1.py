"""Synthetic JSON/CSV/image fixtures; no real scientific assets are opened."""
from pathlib import Path
import copy
import csv
import json
import tempfile
import unittest
from unittest.mock import patch

import audit_shrinkage_matched_native_report_v1 as audit
from test_make_shrinkage_matched_native_report_v1 import synthetic_values, synthetic_gate_files
import make_shrinkage_matched_native_report_v1 as fixture_writer


def write(path, value): Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def record(path): return {"path": str(Path(path).resolve()), "sha256": audit.sha(path)}


def fixture(folder):
    inputs = folder / "inputs"; inputs.mkdir()
    values = synthetic_values(inputs)
    paths, failure = synthetic_gate_files(inputs, values)
    values = {name: audit.read(path) for name, path in paths.items()}
    values["failed_matched_cost_v1"] = audit.read(failure)
    bindings = {name: record(path) for name, path in paths.items()}
    cost = values["matched_cost"]
    bindings.update(failed_matched_cost_v1=record(failure), failed_cost_identity=cost["input_schema_failure_identity"], physical_bridge_repair_addendum=cost["repair_addendum"])
    failed_audit = inputs / "FAILED_AUDIT.json"; write(failed_audit, {"status": "FAILED_PRESERVED", "synthetic": True})
    invocation = inputs / "INVOCATION_IDENTITY.json"
    write(invocation, {"status": "CONFIRMED_PRESERVED_AUDITOR_INVOCATION_PATH_FAILURE", "scientific_inputs_sources_or_outputs_changed": False, "new_F_native_forward_or_training_or_ridge_fit": False, "new_cost_timing": False, "audit_source": values["matched_cost_audit"]["audit_source"], "failed_audit": record(failed_audit)})
    plan = inputs / "REPEAT_PLAN.md"; plan.write_text("Same protocol, preserve every cohort and no favorable selection.\n", encoding="utf-8")
    bindings.update(audit_invocation_failure=record(invocation), cost_repeat_plan=record(plan))
    target = folder / "report"; target.mkdir(); (target / "tables").mkdir(); (target / "figures").mkdir()
    # The fixture producer and independent audit use separate table implementations.
    tables = fixture_writer.collect_tables(values)
    for name, rows in tables.items(): fixture_writer.write_csv(target / "tables" / (name + ".csv"), rows)
    generator = inputs / "make_shrinkage_matched_native_report_v1.py"
    generator.write_bytes((audit.HERE / generator.name).read_bytes())
    prior = inputs / "PRIOR.md"; prior.write_text("Preserved earlier synthetic report.\n", encoding="utf-8")
    linked = {**paths, "failed cost": failure, "schema identity": Path(bindings["failed_cost_identity"]["path"]), "repair": Path(bindings["physical_bridge_repair_addendum"]["path"]), "repeat plan": plan, "invocation failure": invocation, "failed audit": failed_audit, "prior": prior, "report source": generator}
    text = fixture_writer.report_text(values, tables, linked, target)
    (target / "REPORT_KO.md").write_text(text, encoding="utf-8")
    from PIL import Image
    pictures = []
    for stem in audit.FIGURES:
        for suffix in ("png", "svg"):
            path = target / "figures" / (stem + "." + suffix)
            if suffix == "png": Image.new("RGB", (2, 2), (5, 20, 30)).save(path)
            else: path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2"><rect width="2" height="2"/></svg>', encoding="utf-8")
            pictures.append(path.relative_to(target).as_posix())
    manifest = {"status": "COMPLETE", "source": record(generator), "inputs": bindings, "source_bindings": {name: record(path) for name, path in linked.items()}, "prior_reports": [record(prior)], "metadata_only": True,
        **{name: False for name in ("NPZ_or_model_opened", "F_head_fit_or_timer_executed_by_report_generator", "protected_TEST_accessed", "same_decoder_side_information_claimed", "strict_K_only_compression", "independent_generalization_claimed", "Bolt_necessity_proven", "new_rho_seed_checkpoint_or_method_selection", "MAE_or_physical_MSE_minimum_claimed", "strict_FP32_universal_output_lower_bound_claimed", "RAW_bound_applied", "automatic_commit_push", "favorable_cost_cohort_selected")},
        "selected_F_only_cost_cells": 20, "selected_F_only_cost_samples": 120, "new_matched_whole_cost_cells": 28, "new_matched_whole_cost_samples": 196, "repeat_matched_whole_cost_cells": 28, "repeat_matched_whole_cost_samples": 196, "CSV_rows": {name: len(rows) for name, rows in tables.items()}, "figures": pictures,
        "files": [{"path": path.relative_to(target).as_posix(), "sha256": audit.sha(path), "bytes": path.stat().st_size} for path in target.rglob("*") if path.is_file()]}
    write(target / "MANIFEST.json", manifest)
    return target


def rehash(folder, path):
    value = audit.read(folder / "MANIFEST.json")
    row = next(item for item in value["files"] if item["path"] == path.relative_to(folder).as_posix())
    row.update(sha256=audit.sha(path), bytes=path.stat().st_size); write(folder / "MANIFEST.json", value)


def mutate_csv(path, callback):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle); rows, fields = list(reader), reader.fieldnames
    callback(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


class IndependentCombinedReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.folder = fixture(Path(self.temporary.name))

    def test_full_independent_22csv_12images_fixture_without_generator_or_science_import(self):
        original = __import__
        def forbid(name, *args, **kwargs):
            if name.startswith("make_shrinkage_matched_native_report") or name.split(".")[0] in ("numpy", "torch", "chronos"): raise AssertionError("Forbidden scientific/generator import")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=forbid): receipt = audit.verify(self.folder, fixed_sha={})
        self.assertEqual(receipt["CSV_rows"], audit.COUNTS)
        self.assertEqual((receipt["files_checked"], receipt["PNG_count"], receipt["SVG_count"]), (35, 6, 6))
        self.assertGreater(receipt["total_CSV_primitive_fields_checked"], 200000)
        self.assertGreater(receipt["MD_numeric_paragraphs_and_cost_rows_checked"], 50)

    def test_csv_original_derived_cost_ratio_and_adoption_rejected_even_after_rehash(self):
        manifest = audit.read(self.folder / "MANIFEST.json"); tables = audit.expected_tables(audit.terminal(manifest, {}))
        mutations = {"SCORES": ("mse", "999."), "CURRENT_C_PERSISTENCE": ("mse", "999."), "SELECTED_D_WINDOWS": ("lambda_summary", '{"min":999,"median":999,"max":999,"mean":999}'), "OUTPUT_DIAGNOSTIC": ("MAE_is_minimum", "True"), "MATCHED_WHOLE_COST": ("ratio_of_RAW_medians", "999."), "REPEAT_MATCHED_WHOLE_COST": ("median_paired_RAW_ratio", "999."), "REGISTERED_ADOPTION": ("all_conditions_pass", "False")}
        for name, (field, replacement) in mutations.items():
            path = self.folder / "tables" / (name + ".csv"); before = path.read_bytes()
            mutate_csv(path, lambda rows: rows[0].update({field: replacement})); rehash(self.folder, path)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "CSV_SOURCE_VALUE_MISMATCH"): audit.check_csv(path, name, tables[name])
            path.write_bytes(before); rehash(self.folder, path)

    def test_duplicate_keys_even_with_correct_count_rejected(self):
        manifest = audit.read(self.folder / "MANIFEST.json"); tables = audit.expected_tables(audit.terminal(manifest, {}))
        path = self.folder / "tables/CHECKPOINTS.csv"
        mutate_csv(path, lambda rows: rows.__setitem__(1, dict(rows[0])))
        with self.assertRaisesRegex(ValueError, "DUPLICATE_CSV_ROW"): audit.check_csv(path, "CHECKPOINTS", tables["CHECKPOINTS"])

    def test_nested_native_metadata_cannot_hide_wrong_CP(self):
        manifest = audit.read(self.folder / "MANIFEST.json"); tables = audit.expected_tables(audit.terminal(manifest, {}))
        path = self.folder / "tables/NATIVE_TRAINING.csv"
        def change(rows):
            meta = json.loads(rows[0]["metadata"]); meta["selected_update"] = 600; rows[0]["metadata"] = json.dumps(meta)
        mutate_csv(path, change)
        with self.assertRaisesRegex(ValueError, "CSV_SOURCE_VALUE_MISMATCH"): audit.check_csv(path, "NATIVE_TRAINING", tables["NATIVE_TRAINING"])

    def test_three_cost_cohorts_and_timestamp_arithmetic_independently_preserved(self):
        manifest = audit.read(self.folder / "MANIFEST.json"); values = audit.terminal(manifest, {})
        tables = audit.expected_tables(values)
        first, repeated = tables["MATCHED_WHOLE_COST"][0], tables["REPEAT_MATCHED_WHOLE_COST"][0]
        self.assertEqual(repeated["median_seconds"], 2 * first["median_seconds"])
        self.assertEqual(repeated["ratio_of_RAW_medians"], first["ratio_of_RAW_medians"])
        self.assertEqual(len(tables["REGISTERED_ADOPTION"]), 24)
        values["repeat_matched_cost"]["cost"]["robin"]["rows"][0]["median_seconds"] += 1.
        with self.assertRaisesRegex(ValueError, "TIMESTAMP_MEDIAN"): audit.expected_tables(values)

    def test_missing_independent_repeat_PASS_and_bad_counts_rejected(self):
        manifest = audit.read(self.folder / "MANIFEST.json")
        path = Path(manifest["inputs"]["repeat_matched_cost_audit"]["path"])
        original = audit.read(path); value = copy.deepcopy(original); value["status"] = "PENDING"; write(path, value)
        manifest["inputs"]["repeat_matched_cost_audit"] = record(path)
        with self.assertRaisesRegex(ValueError, "COMPLETE_OR_PASS"): audit.terminal(manifest, {})
        value = copy.deepcopy(original); value["totals"]["physical_backend_prechecks_checked"] = 7; write(path, value)
        manifest["inputs"]["repeat_matched_cost_audit"] = record(path)
        with self.assertRaisesRegex(ValueError, "COHORT_COUNTS"): audit.terminal(manifest, {})

    def test_favorable_cohort_selection_scope_rejected(self):
        manifest = audit.read(self.folder / "MANIFEST.json"); manifest["favorable_cost_cohort_selected"] = True; write(self.folder / "MANIFEST.json", manifest)
        with self.assertRaisesRegex(ValueError, "REPORT_SCOPE_CHANGED"): audit.verify(self.folder, fixed_sha={})

    def test_MD_raw_numbers_and_scope_not_only_file_hash(self):
        path = self.folder / "REPORT_KO.md"; text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("정규화 MSE: Raw 1.000000000", "정규화 MSE: Raw 999.000000000", 1), encoding="utf-8"); rehash(self.folder, path)
        with self.assertRaisesRegex(ValueError, "MD_ACTUAL_SCORE"): audit.verify(self.folder, fixed_sha={})

    def test_broken_relative_link_and_wrong_bytes_rejected(self):
        path = self.folder / "REPORT_KO.md"; before = path.read_text(encoding="utf-8")
        path.write_text(before + "\n[missing](tables/MISSING.csv)\n", encoding="utf-8"); rehash(self.folder, path)
        with self.assertRaisesRegex(ValueError, "MISSING_LOCAL_LINK"): audit.verify(self.folder, fixed_sha={})
        path.write_text(before, encoding="utf-8"); rehash(self.folder, path)
        manifest = audit.read(self.folder / "MANIFEST.json"); manifest["files"][0]["bytes"] += 1; write(self.folder / "MANIFEST.json", manifest)
        with self.assertRaisesRegex(ValueError, "ARTIFACT_SHA_OR_BYTES"): audit.verify(self.folder, fixed_sha={})

    def test_no_extra_artifact_and_invalid_image_even_if_rehashed(self):
        extra = self.folder / "EXTRA.csv"; extra.write_text("unregistered\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "NO_UNDECLARED_FILES"): audit.verify(self.folder, fixed_sha={})
        extra.unlink()
        svg = self.folder / "figures" / (audit.FIGURES[0] + ".svg")
        svg.write_text("<xml>fake svg</xml>", encoding="utf-8"); rehash(self.folder, svg)
        with self.assertRaisesRegex(ValueError, "PARSEABLE_SVG"): audit.verify(self.folder, fixed_sha={})

    def test_source_overwrite_refusal_and_maskless_generator_scope(self):
        source = (audit.HERE / "make_shrinkage_matched_native_report_v1.py").read_text(encoding="utf-8-sig")
        self.assertTrue(audit.source_contract(source)["existing_report_directory_refused"])
        bad = source.replace("require(not output.exists(), \"FRESH_INTEGRATED_REPORT_DIRECTORY\")", "require(True, \"FRESH_INTEGRATED_REPORT_DIRECTORY\")")
        with self.assertRaisesRegex(ValueError, "OVERWRITE_REFUSAL"): audit.source_contract(bad)

    def test_audit_output_existing_file_is_refused_without_report_read(self):
        path = self.folder / "ALREADY.json"; path.write_text("preserve me", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "FRESH_REPORT_AUDIT_OUTPUT"): audit.execute(self.folder / "not_opened", path)
        self.assertEqual(path.read_text(encoding="utf-8"), "preserve me")


if __name__ == "__main__": unittest.main()
