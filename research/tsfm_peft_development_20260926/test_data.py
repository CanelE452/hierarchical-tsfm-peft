from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DOC_DIR = ROOT / "research" / "tsfm_peft_development_20260926"
L_CONTEXT = 512
HORIZON = 48
HOUR_NS = 3_600_000_000_000


def load_contract() -> dict:
    return json.loads((DOC_DIR / "data_contract.json").read_text(encoding="utf-8"))


def assert_common_npz(path: Path, origin_keys: tuple[str, ...]) -> None:
    with np.load(path, allow_pickle=False) as data:
        required = {"x", "raw", "mean", "std", "times", "targetmask", "columns", *origin_keys}
        missing = required.difference(data.files)
        assert not missing, f"{path} missing arrays: {sorted(missing)}"
        x = data["x"]
        raw = data["raw"]
        mean = data["mean"]
        std = data["std"]
        times = data["times"]
        targetmask = data["targetmask"]
        columns = data["columns"]

        assert x.dtype == np.float32
        assert raw.dtype == np.float32
        assert mean.dtype == np.float32
        assert std.dtype == np.float32
        assert times.dtype == np.int64
        assert targetmask.dtype == np.bool_
        assert x.shape == raw.shape == targetmask.shape
        assert x.shape[1] == mean.shape[0] == std.shape[0] == columns.shape[0]
        assert np.isfinite(x).all()
        assert np.isfinite(mean).all()
        assert np.isfinite(std).all()
        assert (std > 0).all()
        assert np.all(np.diff(times) > 0)
        assert set(np.unique(np.diff(times))) == {HOUR_NS}

        for key in origin_keys:
            origins = data[key]
            assert origins.dtype == np.int64
            assert (origins >= L_CONTEXT).all(), key
            assert (origins + HORIZON <= x.shape[0]).all(), key
            if origins.size > 1:
                assert np.all(np.diff(origins) > 0), key


def test_data_contract_and_npz_interfaces() -> None:
    contract = load_contract()
    assert contract["context_length"] == L_CONTEXT
    assert contract["horizon"] == HORIZON

    electricity = ROOT / contract["datasets"]["electricity_first32"]["npz"]
    bull = ROOT / contract["datasets"]["bdg2_bull_office"]["npz"]
    assert electricity.exists()
    assert bull.exists()

    assert_common_npz(electricity, ("train_origins", "val_origins", "dev_origins"))
    assert_common_npz(bull, ("train_origins", "val_origins", "eval_e1_origins", "eval_e2_origins"))


def test_electricity_split_origin_bounds() -> None:
    contract = load_contract()
    path = ROOT / contract["datasets"]["electricity_first32"]["npz"]
    with np.load(path, allow_pickle=False) as data:
        split = data["split_bounds"]
        train_end, val_end, total = int(split[1]), int(split[2]), int(split[3])
        assert data["columns"].shape[0] == 32
        assert (data["train_origins"] + HORIZON <= train_end).all()
        assert (data["val_origins"] >= train_end).all()
        assert (data["val_origins"] + HORIZON <= val_end).all()
        assert (data["dev_origins"] >= val_end).all()
        assert (data["dev_origins"] + HORIZON <= total).all()


def test_bull_seal_and_selection_are_train_only_recorded() -> None:
    contract = load_contract()
    bull = contract["datasets"]["bdg2_bull_office"]
    path = ROOT / bull["npz"]
    assert len(bull["sealed_columns_all"]) == 17
    assert bull["selected_count"] == len(bull["selected_columns"])
    assert set(bull["selected_columns"]).issubset(set(bull["sealed_columns_all"]))
    boundary = contract["exposure"]["claim_boundary"].lower()
    assert "do not call" in boundary
    assert "fully independent" in boundary

    with np.load(path, allow_pickle=False) as data:
        assert data["sealed_columns"].shape[0] == 17
        assert data["selected_mask"].shape[0] == 17
        assert int(data["selected_mask"].sum()) == data["columns"].shape[0]
        assert data["raw_all"].shape[1] == 17
        assert data["targetmask_all"].shape == data["raw_all"].shape
        assert data["eval_e1_origins"].size > 0
        assert data["eval_e2_origins"].size > 0


def test_bull_origin_targets_are_inside_declared_half_open_splits() -> None:
    contract = load_contract()
    bull = contract["datasets"]["bdg2_bull_office"]
    path = ROOT / bull["npz"]
    split_to_key = {
        "train": "train_origins",
        "val": "val_origins",
        "eval_e1": "eval_e1_origins",
        "eval_e2": "eval_e2_origins",
    }
    with np.load(path, allow_pickle=False) as data:
        times = data["times"]
        for split, key in split_to_key.items():
            start_s, end_s = bull["split_periods"][split]
            start_ns = pd.Timestamp(start_s).value
            end_ns = pd.Timestamp(end_s).value
            origins = data[key]
            assert origins.size > 0, key
            target_start = times[origins]
            target_end = times[origins + HORIZON - 1] + HOUR_NS
            assert (target_start >= start_ns).all(), key
            assert (target_end <= end_ns).all(), key


def test_normalization_recomputes_from_train_only() -> None:
    contract = load_contract()
    for dataset, train_key in [
        ("electricity_first32", "train"),
        ("bdg2_bull_office", "train"),
    ]:
        path = ROOT / contract["datasets"][dataset]["npz"]
        with np.load(path, allow_pickle=False) as data:
            raw = data["raw"]
            mean = data["mean"]
            std = data["std"]
            x = data["x"]
            if dataset == "electricity_first32":
                start, end = contract["datasets"][dataset]["split_bounds"][train_key]
                train_raw = raw[start:end]
            else:
                times = data["times"]
                start_s, end_s = contract["datasets"][dataset]["split_periods"][train_key]
                idx = (times >= pd.Timestamp(start_s).value) & (times < pd.Timestamp(end_s).value)
                train_raw = raw[idx]
            recomputed_mean = np.nanmean(train_raw, axis=0).astype(np.float32)
            recomputed_std = np.nanstd(train_raw, axis=0).astype(np.float32)
            recomputed_std[~np.isfinite(recomputed_std) | (recomputed_std <= 0)] = np.float32(1.0)
            np.testing.assert_allclose(mean, recomputed_mean, rtol=1e-6, atol=1e-5)
            np.testing.assert_allclose(std, recomputed_std, rtol=1e-6, atol=1e-5)
            assert np.abs(np.nanmean(x[: train_raw.shape[0]], axis=0)).max() < 1e-4 or dataset == "bdg2_bull_office"


def test_bull_protected_raw_changes_do_not_affect_train_only_selection_or_stats() -> None:
    contract = load_contract()
    bull = contract["datasets"]["bdg2_bull_office"]
    path = ROOT / bull["npz"]
    with np.load(path, allow_pickle=False) as data:
        raw = data["raw"].copy()
        times = data["times"]
        columns = data["columns"].copy()
        mean = data["mean"].copy()
        std = data["std"].copy()
        train_start, train_end = bull["split_periods"]["train"]
        train_idx = (times >= pd.Timestamp(train_start).value) & (times < pd.Timestamp(train_end).value)
        protected_idx = ~train_idx
        raw[protected_idx] = np.nan

        recomputed_mean = np.nanmean(raw[train_idx], axis=0).astype(np.float32)
        recomputed_std = np.nanstd(raw[train_idx], axis=0).astype(np.float32)
        recomputed_std[~np.isfinite(recomputed_std) | (recomputed_std <= 0)] = np.float32(1.0)
        np.testing.assert_allclose(mean, recomputed_mean, rtol=1e-6, atol=1e-5)
        np.testing.assert_allclose(std, recomputed_std, rtol=1e-6, atol=1e-5)
        np.testing.assert_array_equal(columns, data["columns"])
