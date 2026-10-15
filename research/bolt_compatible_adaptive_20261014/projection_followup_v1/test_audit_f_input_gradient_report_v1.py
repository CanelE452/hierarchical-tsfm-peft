"""Small generated JSON/JSONL/CSV fixtures; no experiment arrays or generator."""
from pathlib import Path
from unittest.mock import patch
import copy
import csv
import json
import tempfile
import unittest

import audit_f_input_gradient_report_v1 as audit


def score(value, counts=(100, 90, 80)):
    metric = {"mse": value, "mae": value / 2, "per_channel_mse": [value] * 3, "per_channel_mae": [value / 2] * 3, "observed_counts": list(counts), "active_channels": 3}
    return {"normalized": metric, "physical_per_channel_diagnostic": {**metric, "mse": value * 4, "mae": value, "per_channel_mse": [value * 4] * 3, "per_channel_mae": [value] * 3}}


def fixtures(folder):
    asset = folder / "unopened.npz"; asset.write_text("File existence only: this verifier must never open an NPZ", encoding="utf-8")
    binding = {"path": str(asset), "sha256": "0" * 64}
    def entry(value): return {"artifact": binding, "scores_all_observed": score(value), "scores_complete_mask": score(value + .1, (70, 70, 70))}
    values = {"train": {"status": "COMPLETE", "datasets": {}, "resource_terminal": {"cpu_seconds": 2.}}, "parent": {"datasets": {}, "CP0_LS_and_RAW_baselines": {}}, "cost": {"status": "COMPLETE", "cost": {}, "resource_terminal": {"wall_seconds": 1.}}, "audit": {"status": "PASS", "totals": {"stop_gradient_fits_checked": 4}}, "cost_audit": {"status": "PASS", "totals": {"cost_cells_checked": 20}}}
    for data, (_, _, dev) in audit.DATASETS.items():
        values["train"]["datasets"][data] = {"conditions": {}}; values["parent"]["datasets"][data] = {"conditions": {}}
        for seed in audit.SEEDS:
            for label, factor in (("E_LEARN", 1.), ("STOP", 1.2), ("E_FIXED", 1.5)):
                log = folder / (data + label + str(seed) + ".jsonl")
                events = [{"stage": "TRAINING_UPDATE", "update": i + 1, "loss": .2, "E_gradient_norm_before_clip": .3, "D_gradient_norm_before_clip": .4, "global_gradient_norm_before_clip": .5, "clipped": False} for i in range(600)]
                log.write_text("\n".join(json.dumps(r) for r in events), encoding="utf-8")
                cps = [{"update": update, "state": binding, "tune": score(value * factor)["normalized"], **entry(value * factor)} for update, value in zip((0, 100, 200, 400, 600), (3., 2., 1., 1., 2.))]
                row = {"checkpoints": cps, "selected_update": 200, "selected_state": binding, "selection_lock": {"path": str(log), "sha256": audit.record(log)["sha256"]}, "order_sha256": "same", "updates_log": audit.record(log), "global_clipping_fraction": 0., "FIT": entry(factor), "DEV": entry(factor + .2), "CP0_TUNE": cps[0], "RAW_TUNE_scores": score(.5)}
                if label == "STOP":
                    row.update(probe={"status": "PASS", "optimizer_updates": 0, "forward_byte_exact": True, "finite_gradient_vectors": True, "gradient_checks": {"E": {"pass": True, "max_abs": 1e-7}}, "gradient_norms": {"full_E_gradient": .5}, "artifact": binding, "initial_state_sha256": "same"}, CP0_TUNE_byte_checks={"prediction": {"full_stop_byte_exact": True}}, CP0_TUNE_parity_artifact=binding)
                    values["train"]["datasets"][data]["conditions"][str(seed)] = row
                else: values["parent"]["datasets"][data]["conditions"][f"{data}_L0_{label}_s{seed}"] = row
        values["parent"]["CP0_LS_and_RAW_baselines"][data] = {"L0": {split: {"CP0_LS": entry(3.), "RAW_scores": score(.5)} for split in ("fit", dev)}}
        rows = []
        for method in ("RAW_BOLT", "FULL_s92601", "FULL_s92602", "STOP_s92601", "STOP_s92602"):
            for batch in (1, 8):
                duration = (5, 1, 3, 4, 2) if method == "RAW_BOLT" else (4, 2, 6, 3, 1)
                sec = [v / 1000 for v in duration]; raw = [v / 1000 for v in (5, 1, 3, 4, 2)]; ratios = [a/b for a, b in zip(sec, raw)]
                samples = [{"seconds": seconds, "perf_start_ns": 10**18 + i * 10**8, "perf_end_ns": 10**18 + i * 10**8 + value * 10**6} for i, (seconds, value) in enumerate(zip(sec, duration))]
                rows.append({"route": method, "batch": batch, "seconds": sec, "samples": samples, "median_seconds": .003, "median_seconds_per_window": .003/batch, "paired_RAW_ratios": ratios, "paired_RAW_median_ratio": audit.statistics.median(ratios)})
        values["cost"]["cost"][data] = {"rows": rows}
    return values


def write_csv(path, rows):
    names = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names); writer.writeheader()
        for row in rows: writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (dict, list)) else "" if value is None else str(value) for key, value in row.items()})


class StopGradientReportAuditTests(unittest.TestCase):
    def test_independent_metadata_expected_scores_masks_pairing_CP_and_reuse_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            tables = audit.expected_tables(fixtures(Path(folder)))
            self.assertEqual([len(tables[k]) for k in audit.TABLES], [180, 48, 40, 4800, 8, 4, 8, 20, 100, 2, 4])
            row = audit.find(tables["PAIRED"], dataset="robin", seed=92601, split="tune", mask="all_observed", scale="normalized")
            self.assertEqual(row["MSE_stop_minus_full"], 1.2 - 1.); self.assertEqual(row["MAE_stop_minus_full"], .6 - .5)
            complete = audit.find(tables["SCORES"], dataset="robin", seed=92601, method="STOP_GRADIENT", split="tune", mask="complete_C", scale="normalized")
            self.assertEqual(complete["observed_counts"], [70] * 3); self.assertEqual(complete["per_channel_mse"], [1.3] * 3)
            self.assertEqual(sum(r["new_execution"] for r in tables["UPDATES"]), 2400); self.assertEqual(sum(r["selected"] for r in tables["CHECKPOINTS"]), 8)
            self.assertTrue(all(row["selected_update"] == 200 for row in tables["SELECTION"]))

    def test_paired_ratio_differs_from_ratio_of_medians_and_large_ns_remain_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); tables = audit.expected_tables(fixtures(root)); row = audit.find(tables["WHOLE_COST"], dataset="robin", route="STOP_s92601", batch=8)
            self.assertEqual(row["ratio_of_RAW_medians"], 1.); self.assertEqual(row["median_paired_RAW_ratio"], .8)
            expected = tables["WHOLE_COST_SAMPLES"]; path = root / "samples.csv"; write_csv(path, expected)
            self.assertGreater(audit.check_table(path, expected), 100)
            text = path.read_text(); text = text.replace("1000000000000000000", "1000000000000000001", 1); path.write_text(text)
            with self.assertRaises(ValueError): audit.check_table(path, expected)

    def test_changed_cp_order_log_sequence_and_ns_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            values = fixtures(Path(folder)); row = values["train"]["datasets"]["robin"]["conditions"]["92601"]
            bad = copy.deepcopy(values); bad["train"]["datasets"]["robin"]["conditions"]["92601"]["selected_update"] = 400
            with self.assertRaisesRegex(ValueError, "EARLIEST"): audit.expected_tables(bad)
            bad = copy.deepcopy(values); bad["train"]["datasets"]["robin"]["conditions"]["92601"]["order_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "BATCH_ORDER"): audit.expected_tables(bad)
            bad = copy.deepcopy(values); bad["cost"]["cost"]["robin"]["rows"][0]["samples"][0]["perf_end_ns"] += 1
            with self.assertRaisesRegex(ValueError, "INTEGER_NS"): audit.expected_tables(bad)
            path = Path(row["updates_log"]["path"]); lines = path.read_text().splitlines(); lines[0] = lines[1]; path.write_text("\n".join(lines)); row["updates_log"] = audit.record(path)
            with self.assertRaisesRegex(ValueError, "LOGGED_UPDATES"): audit.expected_tables(values)

    def test_all_expected_CSV_primary_fields_and_relative_paths_roundtrip_and_perchannel_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); tables = audit.expected_tables(fixtures(root))
            for name, rows in tables.items():
                expected = audit.relative_values(rows, root); path = root / (name + ".csv"); write_csv(path, expected)
                self.assertGreater(audit.check_table(path, expected), 0); audit.common.check_reference_paths(expected, root)
            path = root / "SCORES.csv"; expected = audit.relative_values(tables["SCORES"], root)
            wrong = copy.deepcopy(expected); wrong[0]["per_channel_mse"][0] += .01; write_csv(path, wrong)
            with self.assertRaises(ValueError): audit.check_table(path, expected)

    def test_closed_actual_gate_does_not_open_reports_or_scientific_files(self):
        with patch.object(audit, "REPORT_SHA", None), patch.object(audit, "record", side_effect=AssertionError("file opened")), self.assertRaisesRegex(ValueError, "NOT_YET_REGISTERED"): audit.verify("missing")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "unopened.npz"; path.write_text("must not read")
            with self.assertRaisesRegex(ValueError, "METADATA_ONLY"): audit.metadata({"path": str(path), "sha256": "0" * 64})

    def test_markdown_four_matched_CP_numbers_ten_cost_numbers_counts_scope_and_links(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); tables = audit.expected_tables(fixtures(root)); (root / "tables").mkdir()
            text = ["TUNE 4/4, 기노출 DEV 4/4, Raw보다 DEV MSE가 작은 조건은 0/8"]
            for data, (_, _, dev) in audit.DATASETS.items():
                label = "Robin" if data == "robin" else "Beijing"
                for seed in audit.SEEDS:
                    t, d = (audit.find(tables["PAIRED"], dataset=data, seed=seed, split=split, mask="all_observed", scale="normalized") for split in ("tune", dev))
                    text.append(f"{label} {seed} 200 200 {t['full_MSE']:.9f} {t['stop_MSE']:.9f} {d['full_MSE']:.9f} {d['stop_MSE']:.9f}")
                for row in tables["WHOLE_COST"]:
                    if row["dataset"] == data and row["batch"] == 8: text.append(f"{label} {row['route']} {row['median_seconds_per_window']*1000:.6f} {row['ratio_of_RAW_medians']:.6f} {row['median_paired_RAW_ratio']:.6f}")
            text += ["F 예측값은 제거하지 않았다; strict K-only 압축이 아니다; 독립 초기화나 독립 표본이 아니다", "실제 F/VJP/backward/optimizer를 재실행한 검사는 아니다; 실제 per-step AdamW parameter update norm을 측정한 것은 아니다", "새 학습은 4×600=2,400 update, 20 CP; 이전 2,400 update·20 CP 로그; probe는 optimizer update 0회; 독립 일반화; 총20조건·100표본"]
            for key, rows in tables.items():
                (root / "tables" / (key + ".csv")).write_text("Exists", encoding="utf-8"); text.append(f"[{key}.csv](tables/{key}.csv): {len(rows)}행")
            original = "\n".join(text); path = root / "REPORT_KO.md"; path.write_text(original, encoding="utf-8")
            self.assertEqual(audit.check_markdown(root, tables), 11)
            for wrong in (original.replace("1.000000000", "1.100000000", 1), original.replace("독립 일반화", "unverified"), original.replace("TUNE 4/4", "TUNE 3/4")):
                path.write_text(wrong, encoding="utf-8")
                with self.assertRaises(ValueError): audit.check_markdown(root, tables)
            path.write_text(original, encoding="utf-8"); (root / "tables/SCORES.csv").unlink()
            with self.assertRaisesRegex(ValueError, "MD_LOCAL_LINK"): audit.check_markdown(root, tables)


if __name__ == "__main__": unittest.main()
