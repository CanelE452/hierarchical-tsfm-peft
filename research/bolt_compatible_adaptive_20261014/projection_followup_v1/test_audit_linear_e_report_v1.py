"""Independent report verifier counterexamples; JSON/CSV and tiny fixture images only."""
import csv
import json
from pathlib import Path

import pytest

import audit_linear_e_report_v1 as audit


def score(value):
    one = {"mse": value, "mae": value / 2, "observed_counts": [48, 24], "active_channels": 2,
           "per_channel_mse": [value, value], "per_channel_mae": [value / 2, value / 2]}
    return {"normalized": one, "physical_per_channel_diagnostic": {**one, "mse": value * 4, "mae": value,
        "per_channel_mse": [value * 4] * 2, "per_channel_mae": [value] * 2}}


def value(value): return {"scores_all_observed": score(value), "scores_complete_mask": score(value + .01)}


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "DATASETS", {"toy": (2, 1, "dev")})
    monkeypatch.setattr(audit, "LAGS", ("L0",)); monkeypatch.setattr(audit, "SEEDS", (92601,))
    t = {"datasets": {"toy": {"conditions": {}}}, "CP0_LS_and_RAW_baselines": {"toy": {"L0": {s: {
        "CP0_LS": value(.9), "RAW_scores": score(.8)} for s in ("fit", "dev")}}}, "failed_paths": []}
    for learn in (False, True):
        name = "toy_L0_" + ("E_LEARN" if learn else "E_FIXED") + "_s92601"
        cps = [{"update": u, "tune": score(.7 if u in (100, 200) else .9)["normalized"],
                "state": {"path": str(tmp_path / (name + str(u) + ".npz")), "sha256": str(u)},
                "parameter_diagnostics": {"E_singularvalues": [1.], "D_feature_to_C_operator_gain": 2.},
                "TUNE_latent_variance": [1.], **value(.7 if u in (100, 200) else .9)} for u in (0, 100, 200, 400, 600)]
        log = tmp_path / (name + ".jsonl")
        log.write_text("".join(json.dumps({"stage": "TRAINING_UPDATE", "update": i, "loss": .3,
            "E_gradient_norm_before_clip": .2 if learn else 0., "D_gradient_norm_before_clip": .3,
            "global_gradient_norm_before_clip": .4, "clipped": False}) + "\n" for i in range(1, 601)), encoding="utf-8")
        row = {"status": "COMPLETE", "lag": "L0", "seed": 92601, "learn_E": learn, "checkpoints": cps,
            "selected_update": 100, "selected_tune": cps[1]["tune"], "selected_state": cps[1]["state"],
            "updates_log": audit.record(log), "global_clipping_fraction": 0., "CP0_TUNE": cps[0], "FIT": value(.6),
            "DEV": value(.7 if learn else .85), "RAW_TUNE_scores": score(.8), "FIT_channel_observed_counts": [48, 24]}
        t["datasets"]["toy"]["conditions"][name] = row
    rows = []
    for batch in (1, 8):
        for route in ("RAW_BOLT", "L0_E_FIXED_s92601", "L0_E_LEARN_s92601"):
            ns = [10000000 + i * 1000000 for i in range(9)]
            if route != "RAW_BOLT": ns = [n // 2 for n in ns]
            samples = [{"repeat": i, "position": i, "perf_start_ns": 1, "perf_end_ns": 1 + n, "seconds": n / 1e9} for i, n in enumerate(ns)]
            sec = [item["seconds"] for item in samples]
            rows.append({"route": route, "batch": batch, "seconds": sec, "samples": samples,
                "median_seconds": sec[4], "median_seconds_per_window": sec[4] / batch,
                "paired_RAW_ratios": [1. if route == "RAW_BOLT" else .5] * 9,
                "paired_RAW_median_ratio": 1. if route == "RAW_BOLT" else .5})
    return {"train": t, "probe": {"cases": {"toy": {"L0": {"status": "PASS", "gradient_diagnostics": {"finite": True}}}}, "failed_paths": []},
        "audit": {"datasets": {}}, "cost": {"cost": {"toy": {"rows": rows}}, "failed_paths": []}, "cost_audit": {}}


def csv_file(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row)) or ["no_failed_paths"]
    with path.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=keys); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def test_actual_report_gate_is_closed_before_paths_are_read(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "REPORT_SHA", None)
    with pytest.raises(ValueError, match="NOT_REGISTERED"):
        audit.verify(tmp_path / "missing")


def test_independent_full_primary_and_derived_tables(tmp_path, monkeypatch):
    values = fixture(tmp_path, monkeypatch); tables = audit.expected_tables(values)
    assert len(tables["SCORES"]) == 42 and len(tables["CHECKPOINTS"]) == 10 and len(tables["UPDATES"]) == 1200
    assert len(tables["PAIRED_E"]) == 12 and len(tables["CHANNEL_COMPARISONS"]) == 6
    assert len(tables["WHOLE_COST"]) == 6 and len(tables["WHOLE_COST_SAMPLES"]) == 54
    for key, rows in tables.items():
        portable = audit.relative_values(rows, tmp_path)
        path = tmp_path / (key + ".csv"); csv_file(path, portable)
        assert audit.check_table(path, portable) > 0 or not rows


def test_copied_channel_metric_tamper_is_detected(tmp_path, monkeypatch):
    tables = audit.expected_tables(fixture(tmp_path, monkeypatch)); rows = tables["SCORES"]
    path = tmp_path / "SCORES.csv"; csv_file(path, rows)
    rows[0]["per_channel_mse"] = [999., 999.]
    csv_file(path, rows)
    expected = audit.expected_tables(fixture(tmp_path, monkeypatch))["SCORES"]
    with pytest.raises(ValueError, match="CSV_CELL"):
        audit.check_table(path, expected)


def test_earliest_exact_tie_is_required(tmp_path, monkeypatch):
    values = fixture(tmp_path, monkeypatch)
    next(iter(values["train"]["datasets"]["toy"]["conditions"].values()))["selected_update"] = 200
    with pytest.raises(ValueError, match="CHECKPOINT_SELECTION"):
        audit.expected_tables(values)


def test_ns_and_ratio_tampering_are_detected(tmp_path, monkeypatch):
    values = fixture(tmp_path, monkeypatch)
    values["cost"]["cost"]["toy"]["rows"][0]["samples"][0]["perf_end_ns"] += 1000
    with pytest.raises(ValueError, match="TIMER_NS"):
        audit.expected_tables(values)
    values = fixture(tmp_path, monkeypatch)
    values["cost"]["cost"]["toy"]["rows"][1]["paired_RAW_median_ratio"] = .4
    with pytest.raises(ValueError, match="PAIRED_RATIO"):
        audit.expected_tables(values)


def test_nonfinite_or_incomplete_update_log_is_rejected(tmp_path, monkeypatch):
    values = fixture(tmp_path, monkeypatch)
    row = next(iter(values["train"]["datasets"]["toy"]["conditions"].values()))
    log = Path(row["updates_log"]["path"]); log.write_text('{"update":1,"loss":NaN}\n', encoding="utf-8")
    row["updates_log"] = audit.record(log)
    with pytest.raises(ValueError, match="NONFINITE"):
        audit.expected_tables(values)


def test_relative_paths_retain_location_and_reject_absolute_CSV_injection(tmp_path):
    source = {"state": {"path": str(tmp_path.parent / "model.npz"), "sha256": "opaque"}}
    expected = audit.relative_values(source, tmp_path)
    assert expected["state"]["path"] == "../model.npz"
    path = tmp_path / "sample.csv"; csv_file(path, [source])
    with pytest.raises(ValueError, match="CSV_CELL"):
        audit.check_table(path, [expected])


def test_image_parsing_and_manifest_path_escape(tmp_path):
    from PIL import Image
    image = tmp_path / "one.png"; Image.new("RGB", (2, 2)).save(image)
    svg = tmp_path / "one.svg"; svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
    assert audit.inspect_images([image, svg]) == (1, 1)
    svg.write_text('<not-svg/>', encoding="utf-8")
    with pytest.raises(ValueError, match="SVG"):
        audit.inspect_images([svg])
    with pytest.raises(ValueError, match="CONTAINMENT"):
        audit.check_manifest_files(tmp_path, [{"path": "../escape.md", "sha256": "bad", "bytes": 0}])


def test_bool_spelling_and_missing_nested_none_are_not_silently_accepted(tmp_path):
    path = tmp_path / "BOOL.csv"; path.write_text("clipped\nbanana\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV_CELL"): audit.check_table(path, [{"clipped": False}])
    with pytest.raises(ValueError): audit.equal({}, {"missing_measurement": None}, "CSV_CELL")


def test_relative_artifact_existence_is_stat_only(tmp_path):
    artifact = tmp_path / "opaque.npz"; artifact.write_bytes(b"not an NPZ and never parsed")
    assert audit.check_reference_paths({"state": {"path": "opaque.npz", "sha256": "opaque"}}, tmp_path) == 1
    with pytest.raises(ValueError, match="RELATIVE_EXISTING"):
        audit.check_reference_paths({"state": {"path": "missing.npz"}}, tmp_path)


def test_accuracy_cost_uses_masked_report_score_not_native_selection_reduction(tmp_path, monkeypatch):
    values = fixture(tmp_path, monkeypatch)
    row = next(iter(values["train"]["datasets"]["toy"]["conditions"].values()))
    row["checkpoints"][1]["scores_all_observed"]["normalized"]["mse"] = .7100000000000001
    tables = audit.expected_tables(values)
    actual = tables["ACCURACY_COST_REFERENCE"][0]["TUNE_MSE_over_RAW"]
    assert actual == .7100000000000001 / .8 and actual != row["selected_tune"]["mse"] / .8


def test_nanosecond_CSV_integers_roundtrip_above_2_to_53_exactly(tmp_path):
    expected = [{"unix_start_ns": 1790000000000000001, "mse": .12345678901234567}]
    path = tmp_path / "NS.csv"; csv_file(path, expected)
    assert audit.check_table(path, expected) == 2
    path.write_text("unix_start_ns,mse\n1790000000000000002,.12345678901234567\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV_CELL"):
        audit.check_table(path, expected)
