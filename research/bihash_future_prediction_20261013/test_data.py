"""CPU-only data contract tests for the BiHash future-prediction campaign."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import data


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def method_lock(tmp_path: Path, dataset: str, source: Path, k: int = 3) -> Path:
    return write_json(
        tmp_path / "METHOD_LOCK.json",
        {
            "locked": True,
            "created_unix": 1.0,
            "K": {data.canonical_dataset(dataset): k},
            "newdata_sources": {data.canonical_dataset(dataset): str(source)},
        },
    )


def joint_method_lock(tmp_path: Path, household_source: Path, sml_source: Path) -> Path:
    return write_json(
        tmp_path / "METHOD_LOCK.json",
        {
            "locked": True,
            "created_unix": 1.0,
            "K": {"uci235_household_hourly": 3, "uci274_sml2010": 5},
            "newdata_sources": {
                "uci235_household_hourly": str(household_source),
                "uci274_sml2010": str(sml_source),
            },
        },
    )


def selected_lock(
    tmp_path: Path,
    fields: dict,
    *,
    extra_fields: list[dict] | tuple[dict, ...] = (),
    selected: bool = True,
    bad_model_sha: bool = False,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifact = tmp_path / "selected_model.bin"
    artifact.write_bytes(b"selected-model")
    digest = sha(artifact)
    if bad_model_sha:
        digest = "0" * 64
    all_fields = [fields, *list(extra_fields)]
    return write_json(
        tmp_path / "GLOBAL_MODEL_LOCK.json",
        {
            "ALL_SELECTED": selected,
            "created_unix": 2.0,
            "newdata_contracts": {f["dataset"]: f for f in all_fields},
            "selected_models": [{"path": str(artifact), "sha256": digest}],
            "controls": [{"path": str(artifact), "sha256": digest}],
            "readouts": [{"path": str(artifact), "sha256": digest}],
        },
    )


def household_raw(tmp_path: Path, hours: int = 1400) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = []
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=hours * 60, freq="min")
    for i, ts in enumerate(timestamps):
        hour = i // 60
        values = [
            1.0 + hour * 0.01,
            0.1 + (i % 7) * 0.01,
            220.0 + (i % 5),
            4.0 + (i % 3),
            float(hour % 4),
            float(hour % 5),
            float(hour % 6),
        ]
        if hour == 100:
            values[2] = np.nan
        rendered = ["?" if not np.isfinite(v) else f"{v:.6f}" for v in values]
        rows.append([ts.strftime("%d/%m/%Y"), ts.strftime("%H:%M:%S"), *rendered])
    path = tmp_path / "household_power_consumption.txt"
    columns = ["Date", "Time", *data.HOUSEHOLD_CHANNELS]
    with path.open("w", encoding="utf-8") as f:
        f.write(";".join(columns) + "\n")
        for row in rows:
            f.write(";".join(row) + "\n")
    return path


def sml_dir(tmp_path: Path, rows: int = 1200, *, conflict_duplicate: bool = False) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    folder = tmp_path / "sml"
    folder.mkdir()
    timestamps = pd.date_range("2012-03-01 00:00:00", periods=rows, freq="15min", tz="UTC")
    records = []
    for i, ts in enumerate(timestamps):
        if i == 700:
            continue
        values = [float(i + j / 10.0) for j in range(21)]
        day = float(ts.dayofweek + 1)
        records.append([ts.strftime("%Y-%m-%d"), ts.strftime("%H:%M:%S"), *values, day])
    duplicate = records[100].copy()
    if conflict_duplicate:
        duplicate[5] = str(float(duplicate[5]) + 1.0)
    records.insert(101, duplicate)
    first, second = records[: len(records) // 2], records[len(records) // 2 :]
    for name, part in (("NEW-DATA-1.T15.txt", first), ("NEW-DATA-2.T15.txt", second)):
        with (folder / name).open("w", encoding="utf-8") as f:
            f.write("# synthetic SML2010 fixture\n")
            for row in part:
                f.write(" ".join(map(str, row)) + "\n")
    return folder


def test_window_forward_fill_resets_to_fit_mean():
    physical = np.array([[np.nan, 3.0], [14.0, np.nan], [np.nan, 9.0]], dtype=np.float32)
    mask = np.isfinite(physical)
    mean = np.array([10.0, 5.0], dtype=np.float32)
    std = np.array([2.0, 2.0], dtype=np.float32)
    normalized = data.window_forward_fill_normalize(physical, mask, mean, std)
    np.testing.assert_allclose(normalized[:, 0], np.array([0.0, 2.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(normalized[:, 1], np.array([-1.0, -1.0, 2.0], dtype=np.float32))


def test_legal_origin_observability_filter_requires_each_channel_future_observation():
    n = data.L + data.H + 5
    observed = np.ones((n, 2), dtype=bool)
    first_origin = data.L
    observed[first_origin:first_origin + data.H, 1] = False
    all_legal = data._legal_origins((0, n), n)
    filtered = data._legal_origins((0, n), n, future_observed=observed)
    assert first_origin in all_legal
    assert first_origin not in filtered
    assert first_origin + 1 in filtered


def test_household_hourly_contract_and_training_payload_excludes_eval_target(tmp_path, monkeypatch):
    source = household_raw(tmp_path)
    lock = method_lock(tmp_path, "uci235", source, k=3)
    contract = data.prepare_new("uci235", lock, cache_root=tmp_path / "cache")

    assert contract["dataset"] == "uci235_household_hourly"
    assert contract["C"] == 7
    assert contract["K"] == 3
    assert contract["parser_metadata"]["hour_observed_rule"].startswith("channel observed")
    assert contract["origins"]["fit"]["count"] == data.FIT_ORIGINS
    assert contract["origins"]["tune"]["count"] == data.TUNE_ORIGINS
    assert contract["origin_observability_filter"]["fit_legal_after_filter"] <= contract["origin_observability_filter"]["fit_legal_before_filter"]

    target_path = Path(contract["payloads"]["eval_future_target"]["path"])
    target_path.write_bytes(b"tampered eval target")
    real_np_load = data.np.load

    def guarded_np_load(path, *args, **kwargs):
        if "EVAL_future_target" in str(path):
            raise AssertionError("training path must not np.load EVAL target")
        return real_np_load(path, *args, **kwargs)

    monkeypatch.setattr(data.np, "load", guarded_np_load)
    loaded, history, target, observed, history_observed = data.load_training_new("uci235", lock, cache_root=tmp_path / "cache")
    fields = data.required_model_selection_fields("uci235", lock, cache_root=tmp_path / "cache")

    assert loaded["eval_target_loaded"] is False
    assert set(loaded["origins"]) == {"fit", "tune"}
    assert loaded["raw_history"]["fit"].shape == history["fit"].shape
    assert history["fit"].dtype == np.float32
    assert target["fit"].dtype == np.float32
    assert observed["fit"].dtype == np.bool_
    assert history_observed["tune"].dtype == np.bool_
    assert "pca_mean" in loaded and "init_mean" in loaded and "basis" in loaded
    assert fields["eval_target_payload_sha256"] == contract["payloads"]["eval_future_target"]["sha256"]

    monkeypatch.setattr(data.np, "load", real_np_load)
    with pytest.raises(ValueError, match="joint fresh data contracts"):
        data.open_new_evaluation("uci235", selected_lock(tmp_path, fields), lock, cache_root=tmp_path / "cache")


def test_sml2010_gap_duplicate_contract_and_eval_open(tmp_path):
    household_source = household_raw(tmp_path / "household_fixture")
    source = sml_dir(tmp_path)
    lock = joint_method_lock(tmp_path, household_source, source)
    data.prepare_new("uci235", lock, cache_root=tmp_path / "cache")
    contract = data.prepare_new("sml2010", lock, cache_root=tmp_path / "cache")
    assert contract["dataset"] == "uci274_sml2010"
    assert contract["C"] == 21
    assert contract["K"] == 5
    assert contract["parser_metadata"]["identical_duplicates_collapsed"] == 1
    assert contract["parser_metadata"]["gap_rows_inserted_as_nan_mask_false"] == 1
    assert "Day_of_week" not in contract["columns"]
    assert "Date" not in contract["columns"]
    assert "Time" not in contract["columns"]

    loaded, history, target, observed, _history_observed = data.load_training_new("sml2010", lock, cache_root=tmp_path / "cache")
    assert loaded["C"] == 21
    assert loaded["basis"].shape == (21, 5)
    assert len(loaded["bank"]["bank_origins"]) <= data.BANK_ORIGINS
    eligibility = data.causal_bank_eligibility(loaded["origins"]["tune"][:2], loaded["bank"]["bank_origins"])
    assert eligibility.shape == (2, data.L, len(loaded["bank"]["bank_origins"]))
    assert target["tune"].shape[1:] == (data.H, 21)
    assert observed["tune"].shape == target["tune"].shape

    fields_household = data.required_model_selection_fields("uci235", lock, cache_root=tmp_path / "cache")
    fields = data.required_model_selection_fields("sml2010", lock, cache_root=tmp_path / "cache")
    with pytest.raises(ValueError, match="ALL_SELECTED"):
        data.open_new_evaluation(
            "sml2010",
            selected_lock(tmp_path / "not_selected", fields, extra_fields=[fields_household], selected=False),
            lock,
            cache_root=tmp_path / "cache",
        )
    opened = data.open_new_evaluation(
        "sml2010",
        selected_lock(tmp_path / "selected", fields, extra_fields=[fields_household]),
        lock,
        cache_root=tmp_path / "cache",
    )
    assert opened["history"].shape[1:] == (data.L, 21)
    assert opened["target"].shape[1:] == (data.H, 21)
    assert np.all(np.diff(opened["origins"]) >= data.EVAL_STRIDE)
    for raw, mask, normalized in zip(opened["physical_history"], opened["history_observed"], opened["history"]):
        rebuilt = data.window_forward_fill_normalize(raw, mask, opened["mean"], opened["std"])
        np.testing.assert_array_equal(rebuilt, normalized)


def test_sml2010_conflicting_duplicate_fails(tmp_path):
    source = sml_dir(tmp_path, conflict_duplicate=True)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        data._parse_sml2010(source)


def test_model_lock_checks_selected_artifact_hash(tmp_path):
    household_source = household_raw(tmp_path / "household_fixture")
    source = sml_dir(tmp_path)
    lock = joint_method_lock(tmp_path, household_source, source)
    data.prepare_new("uci235", lock, cache_root=tmp_path / "cache")
    data.prepare_new("sml2010", lock, cache_root=tmp_path / "cache")
    fields_household = data.required_model_selection_fields("uci235", lock, cache_root=tmp_path / "cache")
    fields = data.required_model_selection_fields("sml2010", lock, cache_root=tmp_path / "cache")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        data.open_new_evaluation(
            "sml2010",
            selected_lock(tmp_path, fields, extra_fields=[fields_household], bad_model_sha=True),
            lock,
            cache_root=tmp_path / "cache",
        )


def test_legacy_loader_reinitializes_pca_k_from_fit_history_pool():
    if not (data.ROOT / "research/level_matched_learned_ed_v18_20261006/data_contract_jena.json").exists():
        pytest.skip("legacy local contracts are absent")
    jena, *_ = data.load_legacy("jena")
    robin, *_ = data.load_legacy("robin")
    assert jena["C"] == 21 and jena["K"] == 5 and jena["basis"].shape == (21, 5)
    assert robin["C"] == 17 and robin["K"] == 4 and robin["basis"].shape == (17, 4)
    assert "FIT-only mean/std" in jena["contract"]["normalization"]
    assert "ignores legacy NPZ basis" in jena["contract"]["basis"]

