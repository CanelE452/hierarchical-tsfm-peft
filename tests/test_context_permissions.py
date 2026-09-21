import pandas as pd
import pytest

from hier_peft.data import DataFailure, origin_windows, split_origins


def test_origin_windows_keep_context_strictly_before_target():
    window = origin_windows(72, context=48, horizon=12, n_times=100)
    assert window["context_indices"][0] == 24
    assert window["context_indices"][-1] == 71
    assert window["target_indices"][0] == 72
    assert window["target_indices"][-1] == 83
    assert max(window["context_indices"]) < min(window["target_indices"])


def test_origin_windows_reject_missing_context_and_out_of_range_target():
    with pytest.raises(DataFailure, match="fewer than 48"):
        origin_windows(47, context=48, horizon=12, n_times=100)
    with pytest.raises(DataFailure, match="exceeds n_times"):
        origin_windows(95, context=48, horizon=12, n_times=100)


def test_split_origins_use_role_contained_targets_and_disjoint_roles():
    dates = pd.date_range("2000-01-01", periods=180, freq="MS")
    manifest = split_origins(len(dates), dates=dates)

    assert manifest["role_bounds"]["CALIBRATION"]["start"] == 84
    assert manifest["role_bounds"]["VALIDATION"]["start"] == 108
    assert manifest["role_bounds"]["TEST"]["start"] == 132
    assert manifest["origin_counts"] == {
        "TRAIN": 25,
        "CALIBRATION": 13,
        "VALIDATION": 13,
        "TEST": 37,
    }

    occupied = {}
    for role, origins in manifest["origins"].items():
        start = manifest["role_bounds"][role]["start"]
        end = manifest["role_bounds"][role]["end_exclusive"]
        for origin in origins:
            window = origin_windows(origin, n_times=len(dates))
            assert max(window["context_indices"]) < origin
            assert min(window["target_indices"]) >= start
            assert max(window["target_indices"]) < end
            for idx in window["target_indices"]:
                assert occupied.setdefault(idx, role) == role


def test_split_origins_requires_enough_history_for_all_roles():
    with pytest.raises(DataFailure) as error:
        split_origins(155)
    assert error.value.code == "TIME_AXIS_TOO_SHORT_FOR_SCREEN_SPLIT"
