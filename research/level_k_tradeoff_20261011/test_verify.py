from pathlib import Path
import importlib.util

import numpy as np


SPEC = importlib.util.spec_from_file_location("verify", Path(__file__).with_name("verify.py"))
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


def test_grouped_metric_channel_macro():
    pred = np.array([[[1.0, 10.0], [3.0, 20.0]]], dtype=np.float32)
    target = np.array([[[0.0, 10.0], [1.0, 18.0]]], dtype=np.float32)
    mask = np.array([[[True, False], [True, True]]])
    got = verify.metric(pred, target, mask)
    assert abs(got["mse"] - (((1.0 + 4.0) / 2.0 + 4.0) / 2.0)) < 1e-12
    assert abs(got["mae"] - (((1.0 + 2.0) / 2.0 + 2.0) / 2.0)) < 1e-12


def test_future_features_lag_layout():
    hist = np.arange(1 * 3 * 2, dtype=np.float32).reshape(1, 3, 2)
    fut = np.arange(100, 104, dtype=np.float32).reshape(1, 2, 2)
    got = verify.future_features_np(hist, fut, lag=2)
    full = np.concatenate([hist[:, -2:], fut], axis=1)
    expected_t0 = np.concatenate([full[:, 2], full[:, 1], full[:, 0]], axis=1)
    assert got.shape == (1, 2, 6)
    assert np.array_equal(got[:, 0], expected_t0)


def test_projection_close_sign_invariant():
    basis = np.eye(3, 2, dtype=np.float32)
    flipped = basis.copy()
    flipped[:, 1] *= -1
    ok, diff = verify.projection_close(basis, flipped)
    assert ok
    assert diff == 0.0


def test_dedupe_rows():
    rows = [{"a": 1}, {"a": 1}, {"a": 2}]
    assert len(verify.dedupe_rows(rows)) == 2


def test_benchmark_dedupe_detects_conflicting_identity():
    rows = [
        {"dataset": "robin", "adapter": "RAW_F0", "batch": 1, "median_seconds": 1.0},
        {"dataset": "robin", "adapter": "RAW_F0", "batch": 1, "median_seconds": 1.0},
        {"dataset": "robin", "adapter": "RAW_F0", "batch": 1, "median_seconds": 2.0},
    ]
    unique, conflicts = verify.dedupe_benchmark_rows(rows)
    assert len(unique) == 1
    assert conflicts and conflicts[0]["mismatch"] == ["median_seconds"]


def test_static_no_test_allows_guard_name():
    src = """
def load_checked(pr, bundle):
    pr.old_pilot().assert_no_test_origins(bundle, bundle.train_origins)
    pr.old_pilot().assert_no_test_origins(bundle, bundle.val_origins)
"""
    assert verify.static_no_test_violations(src, "guard.py") == []


def test_static_no_test_rejects_real_test_access():
    snippets = [
        "loader(include_test=True)\n",
        "obj.test_origins\n",
        "test_origins = []\n",
        "cfg = {'include_test': True}\n",
        "cfg = {'new_test_calls': 1}\n",
        "origins = obj['test_origins']\n",
    ]
    for src in snippets:
        assert verify.static_no_test_violations(src, "bad.py"), src
