"""Independent synthetic report fixtures; no generator, model, or scientific data."""
from pathlib import Path
import copy
import csv
import json
import tempfile
import unittest
from unittest.mock import patch

import verify_local_past_decoder_report_v1 as check


def write(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def record(path):
    return {"path": str(Path(path).resolve()), "sha256": check.sha(path)}


def score(value, channels):
    return {scale: {"mse": float(value), "mae": float(value / 2), "per_channel_MSE": [float(value)] * channels,
        "per_channel_MAE": [float(value / 2)] * channels, "count_per_channel": [48] * channels} for scale in check.SCALES}


def metadata_fixture(folder):
    refs = {}
    for name in ("producer", "fit_target_audit", "past_constant_diagnostic", "past_constant_audit", "past_constant_whole_cost", "past_constant_whole_cost_audit"):
        path = folder / (name + ".json"); write(path, {"synthetic": True}); refs[name] = record(path)
    artifact = {"path": "never_opened_scientific_asset.npz", "sha256": "metadata-only"}
    cpu = {"status": "COMPLETE", **refs, "datasets": {}, "completed_method_pairs": 36, "completed_backend_batches": 24,
        "completed_window_matrix_fits": 3264, "native_seed_D_reuses": 1632, "completed_channel_solver_calls": 44304,
        "completed_no_observation_fallbacks": 0, "failed_paths": [], "new_E_F_native_head_forward_runs": 0,
        "new_E_F_or_native_head_training": False, "new_backbone_training": False, "future_labels_used_for_local_fit": False,
        "method_checkpoint_threshold_selection_used": False, "inference_time_past_decoder_refitting": True,
        "fit_uses_TUNE_DEV_known_past": True, "same_decoder_side_information_as_static": False,
        "past_C_condition_metadata_added": True, "strict_K_only_compression": False,
        "static_D_or_global_rankK_oracle_reused_as_local_lower_bound": False,
        "original_source_cache_selection_SHA_unchanged": True, "independent_holdout": False, "protected_TEST_accessed": False}
    audit = {"status": "PASS", **refs, "datasets": {}, "protected_TEST_accessed": False, "totals": {
        "dataset_split_cells_checked": 6, "RAW_control_cells_checked": 6, "backend_states_checked": 24,
        "completed_backend_batches_checked": 24, "method_pairs_checked": 36, "window_matrix_fits_checked": 3264,
        "channel_slots_checked": 44304, "native_seed_D_reuses_checked": 1632, "static_prediction_arrays_byte_preserved": 36,
        "local_prediction_arrays_checked": 36, "static_einsum_control_arrays_checked": 36, "cached_future_arrays_byte_preserved": 36,
        "score_field_checks": 2592, "new_per_origin_output_diagnostic_backend_cells_checked": 24,
        "new_per_origin_output_inequalities_checked": 36, "RAW_bound_inequalities_checked": 0,
        "observed_channel_solver_calls_checked": 44304, "no_observation_fallbacks_checked": 0}}
    cost = {"status": "COMPLETE", "producer": refs["producer"], "fit_target_audit": refs["fit_target_audit"], "cost": {},
        "new_training_updates": 0, "new_backbone_training": False, "F_before": {"frozen": "same"}, "F_after": {"frozen": "same"},
        "routes": list(check.ROUTES), "failed_paths": [], "completed_cost_cells": 20, "recorded_timer_samples": 120,
        "completed_cell_timer_samples": 120, "local_LS_and_implemented_diagnostics_inside_timer": True,
        "native_heads_loaded_or_timed": False, "local_coefficients_cached_inside_timed_pipeline": False,
        "source_cache_selection_SHA_unchanged": True, "original_selected_forecasts_and_locks_unchanged": True,
        "protected_TEST_accessed": False}
    for dataset, (c, k, dev) in check.DATASETS.items():
        cpu["datasets"][dataset] = {"channels": c, "K": k, "splits": {}}
        audit["datasets"][dataset] = {"splits": {}}
        for split, n in (("fit", 256), ("tune", 64), (dev, 30 if dataset == "robin" else 146)):
            cell = {"RAW_scores_all_observed": score(1., c), "RAW_scores_complete_mask": score(1.1, c), "routes": {}}
            audited = {"routes": {}}
            for projection in check.PROJECTIONS:
                cell["routes"][projection], audited["routes"][projection] = {"backends": {}}, {"backends": {}}
                for backend, methods in check.BACKENDS.items():
                    meta = {"windows": n, "singular_relative_cut": 1e-4, "future_labels_used_in_local_decoder_fit": False,
                        "TUNE_or_DEV_threshold_selection": False, "max_abs_decoder_per_window": [2.] * n,
                        "decoder_Frobenius_norm_per_window": [3.] * n, "decoder_operator_norm_per_window": [2.] * n,
                        "true_observed_counts": [[512] * c for _ in range(n)], "centered_slope_ranks": [[k] * c for _ in range(n)],
                        "affine_model_ranks_with_analytic_intercept": [[k + 1] * c for _ in range(n)],
                        "rank_deficient_slope_channels": [[False] * c for _ in range(n)], "observed_count_below_two": [[False] * c for _ in range(n)],
                        "no_observation_current_C_fallback": [[False] * c for _ in range(n)], "retained_condition_number": [[2.] * c for _ in range(n)],
                        "retained_condition_number_valid": [[True] * c for _ in range(n)], "observed_reconstruction_MSE_real_fit": [[.1] * c for _ in range(n)],
                        "observed_reconstruction_MSE_valid": [[True] * c for _ in range(n)],
                        "singularvalues_observed_centered_Z": [[[1. / (index + 1) for index in range(k)] for _ in range(c)] for _ in range(n)]}
                    predictions, checked_methods = {}, {}
                    for method in methods:
                        predictions[method] = {"static_scores_all_observed": score(.9, c), "local_scores_all_observed": score(1.2, c),
                            "static_scores_complete_mask": score(.95, c), "local_scores_complete_mask": score(1.3, c),
                            "static_D_einsum_arithmetic_replay": {"max_abs_vs_original": 1e-8, "byte_exact": False, "scores_all_observed": score(.9, c)}}
                        checked_methods[method] = {"local_scores_complete_mask": score(1.3, c), "measured_FP32_rounding_MSE_allowance": 1e-8,
                            "local_real_minus_new_origin_specific_minimum_normalized_MSE": 1.1,
                            "local_actual_minus_new_origin_specific_minimum_normalized_MSE": 1.1,
                            "new_local_real_arithmetic_scores_complete_mask": score(1.3, c)}
                    branch = {"local_fit_metadata": meta, "local_state_artifact": artifact, "original_state_source": artifact,
                        "cached_method_sources": {method: artifact for method in methods}, "artifact": artifact,
                        "native_cached_past_byte_exact_and_D_shared": backend == "NATIVE_TORCH32", "methods": predictions,
                        "past_reconstruction_deployment_scores_observed_only": score(.1, c)}
                    diag = {"minimum_only_normalized_MSE_in_each_known_D32_real_rowspace": True,
                        "old_static_or_global_rank_K_minimum_reused": False, "rank_D_per_origin": [k] * n,
                        "certificate_per_origin": [{"index": index, "retained_normal_equation_checked": True} for index in range(n)],
                        "scores": score(.2, c), "RAW_scores_same_complete_mask": score(1.1, c)}
                    cell["routes"][projection]["backends"][backend] = branch
                    audited["routes"][projection]["backends"][backend] = {"methods": checked_methods, "new_per_origin_output_space_diagnostic": diag}
            cpu["datasets"][dataset]["splits"][split], audit["datasets"][dataset]["splits"][split] = cell, audited
        rows = []
        for batch in (1, 8):
            raw_seconds = [float(index + 1) for index in range(6)]
            for route_index, route in enumerate(check.ROUTES):
                multiplier = float(route_index + 1)
                seconds = [value * multiplier for value in raw_seconds]
                samples = [{"dataset": dataset, "route": route, "batch": batch, "repeat": index, "position": (route_index + index) % 5,
                    "seconds": value, "perf_start_ns": 1000000, "perf_end_ns": 1000000 + int(value * 1e9),
                    "unix_start_ns": 1000000, "unix_end_ns": 1000000 + int(value * 1e9)} for index, value in enumerate(seconds)]
                rows.append({"route": route, "batch": batch, "repeats": 6, "seconds": seconds,
                    "median_seconds": 3.5 * multiplier, "median_seconds_per_window": 3.5 * multiplier / batch,
                    "paired_RAW_seconds": raw_seconds, "paired_RAW_ratios": [multiplier] * 6, "paired_RAW_median_ratio": multiplier,
                    "origins": list(range(batch)), "memory": [{"shared_process": True}] * 6,
                    "whole_includes": "localLS and implemented diagnostics", "whole_excludes": "load/save outside timer",
                    "memory_scope": "shared process, no exclusive method claim", "selected_metadata": {"synthetic": True},
                    "reference": artifact, "timed_outputs": artifact, "samples": samples})
        cost["cost"][dataset] = {"rows": rows}
    paths = tuple(folder / name for name in ("CPU.json", "CPU_AUDIT.json", "COST.json", "COST_AUDIT.json"))
    for value, name in ((audit, "audit_local_past_decoder_probe_v1.py"),):
        code = folder / name; code.write_text("independent synthetic code", encoding="utf-8"); value["audit_source"] = record(code)
    write(paths[0], cpu)
    audit["diagnostic"] = record(paths[0]); write(paths[1], audit)
    cost.update(local_diagnostic=record(paths[0]), local_audit=record(paths[1])); write(paths[2], cost)
    code = folder / "audit_local_past_decoder_cost_v1.py"; code.write_text("independent cost source", encoding="utf-8")
    cost_audit = {"status": "PASS", "cost": record(paths[2]), "producer": refs["producer"], "fit_target_audit": refs["fit_target_audit"],
        "local_diagnostic": record(paths[0]), "local_audit": record(paths[1]), "audit_source": record(code), "protected_TEST_accessed": False,
        "totals": {"cost_cells_checked": 20, "recorded_timer_samples_checked": 120, "physical_local_prechecks_checked": 4,
            "physical_local_first8_window_fits_checked": 32, "selected_forecast_file_hashes_checked": 36, "canonical_physical_bridges_checked": 2}}
    write(paths[3], cost_audit)
    return paths


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def fixture(folder):
    inputs = folder / "inputs"; inputs.mkdir()
    cpu, audit, cost, cost_audit = metadata_fixture(inputs)
    target = folder / "report"; target.mkdir(); (target / "tables").mkdir(); (target / "figures").mkdir()
    expected = check.expected_tables(check.read(cpu), check.read(audit), check.read(cost))
    for name, rows in expected.items(): write_csv(target / "tables" / (name + ".csv"), rows)
    prior = []
    for index in range(5):
        path = folder / ("prior" + str(index) + ".md"); path.write_text("accepted prior scientific report", encoding="utf-8")
        prior.append((path, check.sha(path)))
    from PIL import Image
    pictures = []
    for stem in check.FIGURES:
        for suffix in ("png", "svg"):
            path = target / "figures" / (stem + "." + suffix)
            if suffix == "png": Image.new("RGB", (2, 2), (5, 10, 15)).save(path)
            else: path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2"><rect width="2" height="2"/></svg>', encoding="utf-8")
            pictures.append(path.relative_to(target).as_posix())
    source = folder / "make_local_past_decoder_report_v1.py"; source.write_text("synthetic generator identity only; not imported", encoding="utf-8")
    report = target / "REPORT_KO.md"
    report.write_text("\n".join(["![figure](figures/" + stem + ".png)" for stem in check.FIGURES]
        + ["[prior](../prior" + str(index) + ".md)" for index in range(5)]
        + ["[table](tables/" + name + ".csv)" for name in expected]), encoding="utf-8")
    manifest = {"status": "COMPLETE", "source": record(source), "CPU_producer": record(cpu), "CPU_audit": record(audit),
        "cost": record(cost), "cost_audit": record(cost_audit), "source_bindings": {"generator": record(source)},
        "prior_reports": [{"path": str(path.resolve()), "sha256": digest} for path, digest in prior],
        "metadata_only": True, "decoder_side_information_changed": True,
        **{flag: False for flag in ("NPZ_or_model_opened", "training_or_F_forward", "protected_TEST_accessed", "strict_K_only_compression",
            "native_reoptimized_for_local_D", "new_native_deployment_cost_measured", "old_static_or_global_K_floor_reused",
            "MAE_or_physical_MSE_minimum_claimed", "RAW_bound_applied", "independent_generalization_claimed")},
        "CSV_rows": {name: len(rows) for name, rows in expected.items()}, "figures": pictures,
        "files": [{"path": path.relative_to(target).as_posix(), "sha256": check.sha(path), "bytes": path.stat().st_size}
            for path in target.rglob("*") if path.is_file()]}
    write(target / "MANIFEST.json", manifest)
    return target, prior


def rehash(folder, path):
    manifest = check.read(folder / "MANIFEST.json")
    entry = next(item for item in manifest["files"] if item["path"] == path.relative_to(folder).as_posix())
    entry.update(sha256=check.sha(path), bytes=path.stat().st_size)
    write(folder / "MANIFEST.json", manifest)


def mutate_csv(path, mutate):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle); rows, fields = list(reader), reader.fieldnames
    mutate(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


class IndependentVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.folder, self.prior = fixture(Path(self.temporary.name))

    def test_full_independent_synthetic_cohort_no_generator_or_scientific_import(self):
        imported, original_import = [], __import__
        def safe_import(name, *args, **kwargs):
            if name.startswith("make_local_past_decoder_report") or name.split(".")[0] in ("numpy", "torch", "chronos", "continuation_runtime_v1"):
                imported.append(name); raise AssertionError("science or generator import forbidden")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=safe_import): receipt = check.verify(self.folder, self.prior)
        self.assertEqual(imported, [])
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["CSV_rows"], check.COUNTS)
        self.assertEqual(receipt["files_checked"], 23)
        self.assertGreater(receipt["total_CSV_primitive_fields_checked"], 800000)
        self.assertEqual((receipt["PNG_count"], receipt["SVG_count"]), (5, 5))

    def test_csv_corruptions_even_after_rehash_are_rejected(self):
        mutations = {"COMPARISON": ("mse", "999."), "SIGNED_EFFECT": ("local_minus_static_MSE", "999."),
            "LOCAL_WINDOW_GEOMETRY": ("D_operator_norm", "999."), "LOCAL_CHANNEL_SOLVERS": ("centered_slope_ranks", "0"),
            "NEW_PER_ORIGIN_DIAGNOSTIC": ("MAE_is_minimum", "True"), "REGISTERED_ADOPTION": ("all_registered_conditions_pass", "True")}
        manifest = check.read(self.folder / "MANIFEST.json")
        cpu, audit, cost, _ = check.terminal(manifest)
        expected = check.expected_tables(cpu, audit, cost)
        for table, (key, value) in mutations.items():
            path = self.folder / "tables" / (table + ".csv")
            before = path.read_bytes()
            mutate_csv(path, lambda rows: rows[0].update({key: value})); rehash(self.folder, path)
            with self.subTest(table=table), self.assertRaisesRegex(ValueError, "CSV_SOURCE_VALUE_MISMATCH"):
                check.check_csv(path, table, expected[table])
            path.write_bytes(before); rehash(self.folder, path)

    def test_nested_metadata_and_duplicate_keys_are_rejected(self):
        manifest = check.read(self.folder / "MANIFEST.json")
        expected = check.expected_tables(*check.terminal(manifest)[:3])
        path = self.folder / "tables/LOCAL_BACKEND_METADATA.csv"
        def nested(rows):
            value = json.loads(rows[0]["local_fit_metadata"]); value["singular_relative_cut"] = 1e-8
            rows[0]["local_fit_metadata"] = json.dumps(value)
        mutate_csv(path, nested)
        with self.assertRaisesRegex(ValueError, "CSV_SOURCE_VALUE_MISMATCH"): check.check_csv(path, "LOCAL_BACKEND_METADATA", expected["LOCAL_BACKEND_METADATA"])
        path = self.folder / "tables/LOCAL_CHANNEL_SOLVERS.csv"
        mutate_csv(path, lambda rows: rows.__setitem__(1, dict(rows[0])))
        with self.assertRaisesRegex(ValueError, "UNKNOWN_OR_DUPLICATE_CSV_ROW"): check.check_csv(path, "LOCAL_CHANNEL_SOLVERS", expected["LOCAL_CHANNEL_SOLVERS"])

    def test_timer_summary_cannot_override_timestamp_arithmetic(self):
        manifest = check.read(self.folder / "MANIFEST.json")
        cpu, audit, cost, _ = check.terminal(manifest)
        cost["cost"]["robin"]["rows"][0]["median_seconds"] += 1.
        with self.assertRaisesRegex(ValueError, "INDEPENDENT_TIMESTAMP_MEDIAN"): check.expected_tables(cpu, audit, cost)

    def test_complete_cost_without_independent_PASS_is_rejected(self):
        manifest = check.read(self.folder / "MANIFEST.json")
        path = Path(manifest["cost_audit"]["path"])
        value = check.read(path); value["status"] = "PENDING"; write(path, value)
        manifest["cost_audit"] = record(path)
        with self.assertRaisesRegex(ValueError, "COMPLETE_CPU_COST_AND_EXACT_INDEPENDENT_PASSES"): check.terminal(manifest)

    def test_exact_coverage_and_forbidden_future_fit_are_checked(self):
        manifest = check.read(self.folder / "MANIFEST.json")
        path = Path(manifest["cost_audit"]["path"])
        value = check.read(path); value["totals"]["physical_local_first8_window_fits_checked"] = 31; write(path, value)
        manifest["cost_audit"] = record(path)
        with self.assertRaisesRegex(ValueError, "ALL_LOCAL_INDEPENDENT_AUDIT_COUNTS"): check.terminal(manifest)

    def test_future_fit_cannot_be_hidden_by_rebinding_all_SHA_records(self):
        manifest = check.read(self.folder / "MANIFEST.json")
        cpu_path, audit_path, cost_path, cost_audit_path = (Path(manifest[key]["path"]) for key in ("CPU_producer", "CPU_audit", "cost", "cost_audit"))
        cpu, audit, cost, cost_audit = (check.read(path) for path in (cpu_path, audit_path, cost_path, cost_audit_path))
        cpu["future_labels_used_for_local_fit"] = True; write(cpu_path, cpu)
        audit["diagnostic"] = record(cpu_path); write(audit_path, audit)
        cost.update(local_diagnostic=record(cpu_path), local_audit=record(audit_path)); write(cost_path, cost)
        cost_audit.update(cost=record(cost_path), local_diagnostic=record(cpu_path), local_audit=record(audit_path)); write(cost_audit_path, cost_audit)
        for key, path in (("CPU_producer", cpu_path), ("CPU_audit", audit_path), ("cost", cost_path), ("cost_audit", cost_audit_path)):
            manifest[key] = record(path)
        with self.assertRaisesRegex(ValueError, "LOCAL_KNOWN_PAST_FIT_SCOPE"): check.terminal(manifest)

    def test_optional_record_bytes_do_not_change_identity_but_must_be_true(self):
        manifest = check.read(self.folder / "MANIFEST.json")
        entry = manifest["CPU_producer"]
        entry["bytes"] = Path(entry["path"]).stat().st_size
        self.assertEqual(check.terminal(manifest)[0]["status"], "COMPLETE")
        entry["bytes"] += 1
        with self.assertRaisesRegex(ValueError, "EXPECTED_OPTIONAL_BYTES_CHANGED"): check.terminal(manifest)

    def test_bad_report_links_old_version_and_wrong_bytes_rejected(self):
        report = self.folder / "REPORT_KO.md"
        report.write_text(report.read_text(encoding="utf-8") + "\n[missing](tables/missing.csv)", encoding="utf-8")
        rehash(self.folder, report)
        with self.assertRaisesRegex(ValueError, "MISSING_LOCAL_LINK"): check.verify(self.folder, self.prior)
        wrong = list(self.prior); wrong[-1] = (wrong[-1][0], "wrong-version")
        with self.assertRaisesRegex(ValueError, "PRIOR_REPORT_VERSION_CHANGED"): check.verify(self.folder, wrong)
        manifest = check.read(self.folder / "MANIFEST.json"); manifest["files"][0]["bytes"] += 1
        write(self.folder / "MANIFEST.json", manifest)
        with self.assertRaisesRegex(ValueError, "ARTIFACT_SHA_OR_BYTES_CHANGED"): check.verify(self.folder, self.prior)

    def test_manifest_future_generalization_claim_rejected(self):
        manifest = check.read(self.folder / "MANIFEST.json"); manifest["independent_generalization_claimed"] = True
        write(self.folder / "MANIFEST.json", manifest)
        with self.assertRaisesRegex(ValueError, "REPORT_SCOPE_CHANGED"): check.verify(self.folder, self.prior)

    def test_declared_extra_artifact_cannot_hide_outside_exact_image_table_cohort(self):
        path = self.folder / "extra.csv"; path.write_text("extra,unreviewed\n", encoding="utf-8")
        manifest = check.read(self.folder / "MANIFEST.json")
        manifest["files"].append({"path": "extra.csv", "sha256": check.sha(path), "bytes": path.stat().st_size})
        write(self.folder / "MANIFEST.json", manifest)
        with self.assertRaisesRegex(ValueError, "EXACT_TWELVE_TABLES_TEN_IMAGES_ONE_MD"): check.verify(self.folder, self.prior)

    def test_rehashed_invalid_PNG_and_SVG_are_parsed_not_only_hashed(self):
        png = self.folder / "figures" / (check.FIGURES[0] + ".png")
        before = png.read_bytes(); png.write_bytes(b"invalid png"); rehash(self.folder, png)
        with self.assertRaises(Exception): check.verify(self.folder, self.prior)
        png.write_bytes(before); rehash(self.folder, png)
        svg = self.folder / "figures" / (check.FIGURES[0] + ".svg")
        svg.write_text("<xml>not svg</xml>", encoding="utf-8"); rehash(self.folder, svg)
        with self.assertRaisesRegex(ValueError, "PARSEABLE_SVG"): check.verify(self.folder, self.prior)


if __name__ == "__main__":
    unittest.main()
