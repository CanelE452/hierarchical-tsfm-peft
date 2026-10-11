from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pca_basis_uses_centered_train_rows_only():
    sweep = load_module("k_sweep_under_test", "sweep.py")
    train = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    val_with_large_third_axis = np.asarray([[0.0, 0.0, 100.0], [0.0, 0.0, -100.0]], dtype=np.float32)
    x = np.vstack([train, val_with_large_third_axis])

    basis = sweep.compute_pca_basis(x, (0, 4), rank=2)

    expected = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(np.abs(basis), expected, atol=1e-6)


def test_adapter_plan_has_explicit_k_ids_and_raw_c_control():
    sweep = load_module("k_sweep_under_test", "sweep.py")
    protocol = {
        "datasets": ["robin", "jena"],
        "k_grid": {"robin": [2, 5, 10], "jena": [3, 6, 12]},
        "seeds": [92601, 92602],
    }

    robin = sweep.adapter_plan(protocol, "robin")
    jena = sweep.adapter_plan(protocol, "jena")
    all_ids = [row["id"] for row in robin + jena]

    assert len(robin) == 10
    assert len(jena) == 10
    assert len(set(all_ids)) == len(all_ids)
    assert {row["id"] for row in robin if row["family"] == "PCA"} == {"robin_PCA_K2", "robin_PCA_K5", "robin_PCA_K10"}
    assert "robin_RAW_C_D8" in all_ids
    assert "PCA_ANCHOR" not in all_ids


def test_fit_scaled_ridge_grid_keeps_all_coefficients_and_earliest_tie():
    sweep = load_module("k_sweep_under_test", "sweep.py")
    train_features = np.asarray([[[1.0], [2.0]]], dtype=np.float64)
    train_target = np.asarray([[[2.0], [4.0]]], dtype=np.float32)
    train_mask = np.ones_like(train_target, dtype=bool)
    train_row_mask = np.asarray([[True, True]])
    val_features = train_features.copy()
    val_target = train_target.copy()
    val_mask = train_mask.copy()

    fitted = sweep.fit_scaled_ridge_grid(
        train_features,
        train_target,
        train_mask,
        train_row_mask,
        val_features,
        val_target,
        val_mask,
        lambdas=[0.0, 0.0],
    )

    assert fitted["selected_index"] == 0
    assert fitted["selected_lambda"] == 0.0
    assert len(fitted["choices"]) == 2
    assert fitted["grid_coefficients"].shape == (2, 1, 1)
    np.testing.assert_allclose(fitted["coeff"], [[2.0]], atol=1e-12)


def test_future_feature_boundary_uses_history_tail_without_future_truth():
    sweep = load_module("k_sweep_under_test", "sweep.py")
    z_history = np.asarray([[[10.0], [20.0]]], dtype=np.float32)
    z_future = np.asarray([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    features = sweep.future_features(z_history, z_future, lag=2)
    coeff = np.asarray([[100.0], [10.0], [1.0]], dtype=np.float64)

    pred = sweep.apply_coeff(features, coeff)

    np.testing.assert_allclose(pred[:, :, 0], [[310.0, 230.0, 321.0]], atol=1e-6)


def test_benchmark_case_rows_count_k_and_optional_method_cases():
    benchmark = load_module("k_benchmark_under_test", "benchmark.py")
    result = {
        "protocol": {"datasets": ["robin", "jena"], "benchmark": {"batches": [1, 32]}},
        "datasets": {
            "robin": {"benchmark_cases": ["RAW_F0", "robin_RAW_C_D8", "robin_PCA_K2", "robin_MLP_POINT_K2_seed92601"]},
            "jena": {"benchmark_cases": ["RAW_F0", "jena_RAW_C_D8", "jena_PCA_K3", "jena_MLP_POINT_K3_seed92601"]},
        },
    }
    method_result = {
        "datasets": {
            "robin": {"benchmark_cases": ["robin_TS2VEC_K5_seed92601"]},
            "jena": {"benchmark_cases": ["jena_TS2VEC_K6_seed92601"]},
        }
    }

    rows = benchmark.case_rows(result, method_result=method_result)

    assert len(rows) == 20
    assert {row["batch"] for row in rows} == {1, 32}
    assert any(row["adapter"] == "robin_RAW_C_D8" and row["k"] == 17 for row in rows)
    assert any(row["adapter"] == "robin_TS2VEC_K5_seed92601" and row["source"] == "method" for row in rows)


def test_prediction_entry_rejects_path_hash_mismatch(tmp_path):
    benchmark = load_module("k_benchmark_under_test", "benchmark.py")
    npz_path = tmp_path / "pred.npz"
    np.savez_compressed(npz_path, prediction_tsfm=np.zeros((1, 1, 1), dtype=np.float32))
    result = {
        "protocol": {"benchmark": {"contract": "FORECAST_RIDGE_D8"}},
        "datasets": {
            "robin": {
                "raw_f0": {},
                "decoder_contracts": {
                    "robin_PCA_K2": {
                        "contracts": {
                            "FORECAST_RIDGE_D8": {"prediction_npz": str(npz_path), "prediction_sha256": "bad"}
                        }
                    }
                },
            }
        },
    }

    with pytest.raises(RuntimeError, match="prediction NPZ changed"):
        benchmark.load_decoder_coeff(result, "robin", "robin_PCA_K2")


def test_replace_retry_recovers_from_transient_permission_error(tmp_path, monkeypatch):
    sweep = load_module("k_sweep_under_test", "sweep.py")
    tmp = tmp_path / "out.json.tmp"
    target = tmp_path / "out.json"
    tmp.write_text('{"ok": true}\n', encoding="utf-8")
    original_replace = type(tmp).replace
    calls = {"replace": 0}
    sleeps = []

    def flaky_replace(self, other):
        if self == tmp and calls["replace"] < 2:
            calls["replace"] += 1
            raise PermissionError("simulated reader race")
        return original_replace(self, other)

    monkeypatch.setattr(type(tmp), "replace", flaky_replace)
    monkeypatch.setattr(sweep.time, "sleep", lambda seconds: sleeps.append(seconds))

    sweep.replace_with_permission_retry(tmp, target, delays=(0.001, 0.002, 0.003))

    assert calls["replace"] == 2
    assert sleeps == [0.001, 0.002]
    assert target.read_text(encoding="utf-8") == '{"ok": true}\n'


def test_replace_retry_raises_after_retry_cap(tmp_path, monkeypatch):
    benchmark = load_module("k_benchmark_under_test", "benchmark.py")
    tmp = tmp_path / "out.json.tmp"
    target = tmp_path / "out.json"
    tmp.write_text("{}", encoding="utf-8")
    calls = {"replace": 0}
    sleeps = []

    def locked_replace(self, other):
        if self == tmp:
            calls["replace"] += 1
            raise PermissionError("still locked")
        raise AssertionError("unexpected path")

    monkeypatch.setattr(type(tmp), "replace", locked_replace)
    monkeypatch.setattr(benchmark.time, "sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(PermissionError, match="still locked"):
        benchmark.replace_with_permission_retry(tmp, target, delays=(0.001, 0.002))

    assert calls["replace"] == 3
    assert sleeps == [0.001, 0.002]
