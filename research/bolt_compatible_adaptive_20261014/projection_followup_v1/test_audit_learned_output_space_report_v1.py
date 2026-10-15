"""Generated metadata/CSV/Markdown only; no generator or scientific payload reads."""
from pathlib import Path
from unittest.mock import patch
import copy
import csv
import json
import tempfile
import unittest

import audit_learned_output_space_report_v1 as audit


CONDITIONS = tuple(f"FULL_{lag}_s{s}" for lag in ("L0", "L7") for s in (92601, 92602)) + ("STOP_L0_s92601", "STOP_L0_s92602")


def metric(value, count=10):
    unit = {"mse": value, "mae": value / 2, "per_channel_mse": [value] * 3, "per_channel_mae": [value / 2] * 3, "observed_counts": [count] * 3, "active_channels": 3}
    return {"normalized": unit, "physical_per_channel_diagnostic": {**unit, "mse": value * 4, "mae": value, "per_channel_mse": [value * 4] * 3, "per_channel_mae": [value] * 3}}


def fixture(folder):
    asset = folder / "never_open.npz"; asset.write_text("Existence check only", encoding="utf-8"); binding = {"path": str(asset), "sha256": "1"*64, "bytes": asset.stat().st_size, "arrays": {"do_not_open": {}}}
    result = {"status": "COMPLETE", "datasets": {}, "completed_diagnostic_cells": 24, "resource_terminal": {"cpu_seconds": 2., "wall_seconds": 3., "status": "COMPLETE"}}
    for data, (_, _, dev) in audit.DATASETS.items():
        result["datasets"][data] = {"conditions": {}}
        for index, name in enumerate(CONDITIONS):
            owner = {"origin": "FULL_REUSED" if name.startswith("FULL") else "STOP_REUSED", "seed": int(name[-5:]), "selected_update": 400, "selected_state": binding, "selection_lock": binding, "splits": {}}
            for split in ("tune", dev):
                floor = (1.2 if data == "robin" and split == dev else .3) + index * .001; actual, ey, raw = floor + .8, floor + .2, 1.
                base = {"status": "COMPLETE", "complete_target_positions": 20, "target_positions": 24, "joint_unknown_future_coordinates": 48, "joint_output_coordinates_before_mask": 144, "lag_slope_parameters_per_C": 7 if "L7" in name else 1, "minimum_normalized_MSE": floor, "EY_encoding_backend": "NumPy FP32 GEMM", "selected_E_D_previously_supervised_by_FIT_future_targets": True, "EY_incomplete_C_horizons_filled_with_current_C": 4, "EY_fill_can_affect_later_L7_complete_outputs": "L7" in name, "RAW_bound_applied": False, "all_real_latent_or_all_FP32_lower_bound_claimed": False, "methods": {}, "input_source": {**binding, "sha256": data + split}, "selected_forecast_source": binding, "artifact": binding}
                base["joint_metadata"] = {"numeric_rank_per_origin": [2, 2], "rcond": 1e-15, "minimum_space_definition": "registered retained numeric output space"}
                for method, value in (("ACTUAL_F", actual), ("EY", ey), ("FLAT", floor + .6)):
                    base["methods"][method] = {"role": method + "_ROLE", "scores_complete_mask": metric(value), "scores_all_observed": metric(value + .01, 11), "real_trace_scores_complete_C": metric(value + .0001), "retained_output_real_trace_scores_complete_C": metric(value + .0002), "actual_FP32_minus_minimum": value - floor, "real_trace_minus_minimum": value + .0001 - floor, "retained_output_trace_bound_inequality_pass": True, "real_trace_bound_inequality_pass": False, "FP32_bound_with_measured_allowance_pass": True, "all_real_latent_forecasts_bound_claimed": False}
                base.update(RAW_BOLT_scores_complete_mask=metric(raw), RAW_BOLT_scores_all_observed=metric(raw + (.1 if data != "robin" else 0), 11), current_C_persistence_scores_complete_mask=metric(raw + .2), minimum_solution_scores_complete_C=metric(floor))
                cert = {"window_indices": [0, 1], "complete_horizons": [True, False, True], "selected_output_rows": 6, "unknown_future_coordinates_kept": 48, "numeric_rank": 2, "singularvalues": [2., .1, 1e-18], "normal_residual_max_abs": 1e-12, "normal_tolerance": 1e-10, "exact_zero_reduced_LS": {"deliberately_outside_selected_verification_scope": True}}
                base["independent_LS_certificates"] = [cert]; owner["splits"][split] = base
            result["datasets"][data]["conditions"][name] = owner
    return result


def write_csv(path, rows):
    names = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names); writer.writeheader()
        for row in rows: writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict)) else "" if value is None else str(value) for key, value in row.items()})


def markdown(tables, folder):
    lines = ["minimum>Raw가 TUNE 0/6 → 기노출 DEV 6/6", "minimum>Raw 0/6, E(Y)<Raw 6/6, actual F>Raw 6/6", "새 학습·F 추론·새 선정·배포 비용 측정은 모두0; rcond1e−15; 보편 하한도 아니다; MAE·물리 MSE는 각각의 minimum이 아니며; E(Y)는 배포 입력이 아니다; strict K-only도 아니다; 동일 고정 공간의 단독 인과효과로 부르지 않는다; 24개의 독립 평가가 아니다; 과거 E/D의 FIT 미래 감독은 인정; 독립 일반화; 예측 목적 표현의 첫 시도가 아니다"]
    for row in tables["CELLS"]:
        label = "Robin" if row["dataset"] == "robin" else "Beijing"
        lines.append(f"{label} {row['split']} {row['condition']} " + " ".join(f"{row[k]:.9f}" for k in ("minimum_normalized_MSE", "actual_complete_MSE", "EY_complete_MSE", "RAW_complete_MSE", "RAW_all_observed_MSE")))
        if row["condition"] == "FULL_L0_s92601": lines.append(f"대표 FULL_L0/seed92601 {row['split']}: complete-C normalized MSE actual F={row['actual_complete_MSE']:.9f}, E(Y)={row['EY_complete_MSE']:.9f}, minimum={row['minimum_normalized_MSE']:.9f}, Raw={row['RAW_complete_MSE']:.9f}.")
    (folder / "tables").mkdir()
    for key, rows in tables.items():
        (folder / "tables" / (key + ".csv")).write_text("Link exists", encoding="utf-8"); lines.append(f"[{key}.csv](tables/{key}.csv): {len(rows)}행")
    return "\n".join(lines)


class LearnedOutputReportTests(unittest.TestCase):
    def test_24_cell_phase_mask_perchannel_trace_and_matched_difference_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            tables = audit.expected_tables(fixture(Path(folder)))
            self.assertEqual([len(tables[k]) for k in audit.TABLES], [24, 672, 72, 24, 8, 0, 1])
            row = audit.find(tables["SCORES"], dataset="beijing_aotizhongxin", condition="FULL_L0_s92601", split="eval", method="RAW_BOLT", phase="FP32", mask="complete_C", scale="normalized")
            self.assertEqual(row["mse"], 1.); self.assertEqual(row["observed_counts"], [10]*3)
            allobs = audit.find(tables["SCORES"], dataset="beijing_aotizhongxin", condition="FULL_L0_s92601", split="eval", method="RAW_BOLT", phase="FP32", mask="all_observed", scale="normalized")
            self.assertEqual(allobs["mse"], 1.1); self.assertEqual(allobs["observed_counts"], [11]*3)
            delta = audit.find(tables["MATCHED_L0_FULL_STOP"], dataset="robin", seed=92601, split="tune")
            self.assertEqual(delta["minimum_normalized_MSE_stop_minus_full"], .304 - .3)
            self.assertFalse(tables["TRACES"][0]["real_trace_bound_inequality_pass"])

    def test_selected_CSV_fields_roundtrip_and_rank_summary_perchannel_boolean_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); tables = audit.expected_tables(fixture(root))
            for name, rows in tables.items():
                expected = audit.relative_values(rows, root); path = root / (name + ".csv"); write_csv(path, expected)
                audit.check_table(path, expected); audit.common.check_reference_paths(expected, root)
            for name, key in (("CELLS", "RAW_complete_MSE"), ("TRACES", "real_trace_bound_inequality_pass"), ("SVD_GROUPS", "last_retained_singular")):
                expected = audit.relative_values(tables[name], root); bad = copy.deepcopy(expected); bad[0][key] = not bad[0][key] if isinstance(bad[0][key], bool) else bad[0][key] + .01
                path = root / (name + ".csv"); write_csv(path, bad)
                with self.subTest(name=name), self.assertRaises(ValueError): audit.check_table(path, expected)

    def test_24_MD_rows_conditional_counts_representatives_scope_and_relative_links(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); tables = audit.expected_tables(fixture(root)); original = markdown(tables, root); path = root / "REPORT_KO.md"; path.write_text(original, encoding="utf-8")
            self.assertEqual(audit.check_markdown(root, tables), 7)
            for text in (original.replace("기노출 DEV 6/6", "기노출 DEV 5/6"), original.replace("E(Y)<Raw 6/6", "E(Y)<Raw 0/6"), original.replace("2.000000000", "2.100000000", 1), original.replace("독립 일반화", "")):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError): audit.check_markdown(root, tables)
            path.write_text(original, encoding="utf-8"); (root / "tables/SCORES.csv").unlink()
            with self.assertRaisesRegex(ValueError, "EXISTING_MD_LINK"): audit.check_markdown(root, tables)

    def test_failure_and_resource_are_preserved_without_invented_complete_cells(self):
        with tempfile.TemporaryDirectory() as folder:
            result = fixture(Path(folder)); owner = result["datasets"]["robin"]["conditions"]["FULL_L0_s92601"]; owner["splits"]["val"] = {"status": "FAILED_PATH_PRESERVED", "failure": {"message": "synthetic missing artifact"}}; result["completed_diagnostic_cells"] = 23
            tables = audit.expected_tables(result); self.assertEqual(len(tables["CELLS"]), 23); self.assertEqual(len(tables["FAILURES"]), 1)
            self.assertEqual(tables["FAILURES"][0]["failure"]["message"], "synthetic missing artifact")
            row = audit.find(tables["MATCHED_L0_FULL_STOP"], dataset="robin", seed=92601, split="val")
            self.assertEqual(row["status"], "UNAVAILABLE_PARTIAL"); self.assertIsNone(row["minimum_normalized_MSE_stop_minus_full"])
            self.assertEqual(tables["RESOURCE"][0]["wall_seconds"], 3.)

    def test_closed_actual_report_gate_and_metadata_helper_reject_scientific_payload(self):
        with patch.object(audit, "REPORT_SHA", None), patch.object(audit, "record", side_effect=AssertionError("unexpected file read")), self.assertRaisesRegex(ValueError, "NOT_YET_REGISTERED"): audit.verify("missing")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "unopened.npz"; path.write_text("Must not read", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "METADATA_ONLY"): audit.metadata({"path": str(path), "sha256": "0"*64})


if __name__ == "__main__": unittest.main()
