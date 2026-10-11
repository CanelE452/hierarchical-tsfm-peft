from __future__ import annotations

import importlib.util
import json
import sys
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


def test_owned_storage_skips_pycache_and_tmp_races(tmp_path, monkeypatch):
    sweep = load_module("k_sweep_under_test", "sweep.py")
    root = tmp_path / "owned"
    root.mkdir()
    stable = root / "stable.bin"
    stable.write_bytes(b"12345")
    tmp_file = root / "progress.json.1.tmp"
    tmp_file.write_bytes(b"gone")
    pycache = root / "__pycache__"
    pycache.mkdir()
    ignored = pycache / "module.pyc"
    ignored.write_bytes(b"ignored")
    original_stat = sweep.Path.stat

    def racing_stat(self, *args, **kwargs):
        if self == tmp_file:
            raise FileNotFoundError(str(self))
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(sweep.Path, "stat", racing_stat)

    assert sweep.owned_storage_bytes_for((root,)) == 5


def test_owned_storage_raises_non_ephemeral_file_not_found(tmp_path, monkeypatch):
    sweep = load_module("k_sweep_under_test", "sweep.py")
    root = tmp_path / "owned"
    root.mkdir()
    missing = root / "result.json"
    missing.write_text("{}", encoding="utf-8")
    original_stat = sweep.Path.stat

    def racing_stat(self, *args, **kwargs):
        if self == missing:
            raise FileNotFoundError(str(self))
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(sweep.Path, "stat", racing_stat)

    with pytest.raises(FileNotFoundError):
        sweep.owned_storage_bytes_for((root,))


def test_continuation_validates_active_cache_reuse_by_preserved_sha(tmp_path, monkeypatch):
    sys.path.insert(0, str(HERE))
    try:
        continuation = load_module("k_continuation_under_test", "continuation.py")
    finally:
        sys.path.remove(str(HERE))
    active_cache = tmp_path / ".cache" / "level_k_tradeoff_20261011"
    active_file = active_cache / "predictions" / "robin_PCA_K2_FORECAST_RIDGE_D8_val.npz"
    active_file.parent.mkdir(parents=True)
    active_file.write_bytes(b"prediction")
    archived_attempt = tmp_path / "research" / "level_k_tradeoff_20261011" / "failed_attempts" / "attempt2"
    archived_file = archived_attempt / "cache" / "predictions" / active_file.name
    archived_file.parent.mkdir(parents=True)
    archived_file.write_bytes(b"prediction")
    monkeypatch.setattr(continuation, "CACHE", active_cache)
    monkeypatch.setattr(continuation, "ATTEMPT2", archived_attempt)
    manifest = {
        "archived_artifacts": [
            {
                "path": continuation.sweep.public_path(archived_file),
                "sha256": continuation.sweep.sha256_path(archived_file),
                "bytes": archived_file.stat().st_size,
            }
        ]
    }

    receipt = continuation.validate_active_cache_reuse(manifest)

    assert receipt["checked_files"] == 1
    assert receipt["checked_bytes"] == len(b"prediction")


def test_continuation_rejects_unpreserved_active_cache_file(tmp_path, monkeypatch):
    sys.path.insert(0, str(HERE))
    try:
        continuation = load_module("k_continuation_under_test_2", "continuation.py")
    finally:
        sys.path.remove(str(HERE))
    active_cache = tmp_path / ".cache" / "level_k_tradeoff_20261011"
    active_file = active_cache / "curves" / "unexpected.json"
    active_file.parent.mkdir(parents=True)
    active_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(continuation, "CACHE", active_cache)

    with pytest.raises(RuntimeError, match="not attempt2-preserved"):
        continuation.validate_active_cache_reuse({"archived_artifacts": []})


def test_continuation_aggregate_counts_are_derived_from_ledgers(tmp_path, monkeypatch):
    sys.path.insert(0, str(HERE))
    try:
        continuation = load_module("k_continuation_counts_under_test", "continuation.py")
    finally:
        sys.path.remove(str(HERE))
    attempt1 = tmp_path / "attempt1"
    attempt2 = tmp_path / "attempt2"
    attempt1.mkdir()
    attempt2.mkdir()
    (attempt1 / "failure_ledger.json").write_text(
        '{"fit_starts":1,"completed_fits":0,"ridge_fits":0,"wall_seconds":12.0,"process_cpu_seconds":48.0,"failure_progress":{"rss_bytes":10,"cuda_max_allocated_bytes":20,"cuda_max_reserved_bytes":30}}',
        encoding="utf-8",
    )
    (attempt2 / "failure_ledger.json").write_text(
        '{"fit_starts":6,"completed_fits":5,"ridge_fits":100,"wall_seconds":358.0,"process_cpu_seconds":1244.0,"failure_progress":{"rss_bytes":40,"cuda_max_allocated_bytes":5,"cuda_max_reserved_bytes":7}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(continuation, "ATTEMPT1", attempt1)
    monkeypatch.setattr(continuation, "ATTEMPT2", attempt2)

    counts = continuation.aggregate_execution_counts(
        {"fit_starts": 3, "completed_fits": 3, "ridge_fits": 100, "elapsed_seconds": 9.0, "process_cpu_seconds": 11.0}
    )

    assert counts["fit_starts"] == 10
    assert counts["completed_fits"] == 8
    assert counts["ridge_fits"] == 200
    assert counts["continuation_counts"] == {"fit_starts": 3, "completed_fits": 3, "ridge_fits": 100}
    assert counts["stage_wall_seconds_sum"] == 379.0
    assert counts["stage_peak_rss_bytes_max"] == 40


def test_continuation_aggregate_rejects_unexpected_current_counts(tmp_path, monkeypatch):
    sys.path.insert(0, str(HERE))
    try:
        continuation = load_module("k_continuation_counts_under_test_2", "continuation.py")
    finally:
        sys.path.remove(str(HERE))

    with pytest.raises(RuntimeError, match="unexpected_continuation_counts"):
        continuation.aggregate_execution_counts({"fit_starts": 3, "completed_fits": 2, "ridge_fits": 100})


def test_continuation_source_seal_requires_exact_common_six_plus_continuation(tmp_path, monkeypatch):
    sys.path.insert(0, str(HERE))
    try:
        continuation = load_module("k_continuation_seal_under_test", "continuation.py")
    finally:
        sys.path.remove(str(HERE))
    root = tmp_path / "repo"
    here = root / "research" / "level_k_tradeoff_20261011"
    here.mkdir(parents=True)
    names = ["PLAN.md", "protocol.json", "sweep.py", "benchmark.py", "test_sweep.py", "seal.py", "continuation.py"]
    artifacts = []
    for name in names:
        path = here / name
        path.write_text(name, encoding="utf-8")
        artifacts.append({"path": f"research/level_k_tradeoff_20261011/{name}", "sha256": continuation.sweep.sha256_path(path), "bytes": path.stat().st_size})
    seal = here / "source_seal.continuation.json"
    seal.write_text(
        json.dumps({"phase": "before_stage_science", "stage": "continuation", "artifacts": artifacts}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(continuation, "HERE", here)
    monkeypatch.setattr(continuation.sweep, "ROOT", root)

    receipt = continuation.verify_continuation_seal()

    assert len(receipt["artifacts"]) == 7
    assert receipt["path"] == seal


def test_continuation_source_seal_rejects_extra_analysis_file(tmp_path, monkeypatch):
    sys.path.insert(0, str(HERE))
    try:
        continuation = load_module("k_continuation_seal_under_test_2", "continuation.py")
    finally:
        sys.path.remove(str(HERE))
    root = tmp_path / "repo"
    here = root / "research" / "level_k_tradeoff_20261011"
    here.mkdir(parents=True)
    bad = here / "analyze.py"
    bad.write_text("analysis", encoding="utf-8")
    seal = here / "source_seal.continuation.json"
    seal.write_text(
        json.dumps({"phase": "before_stage_science", "stage": "continuation", "artifacts": [{"path": "research/level_k_tradeoff_20261011/analyze.py", "sha256": continuation.sweep.sha256_path(bad)}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(continuation, "HERE", here)
    monkeypatch.setattr(continuation.sweep, "ROOT", root)

    with pytest.raises(RuntimeError, match="continuation_seal_artifacts_mismatch"):
        continuation.verify_continuation_seal()


def test_continuation_clean_inherited_failure_fields():
    sys.path.insert(0, str(HERE))
    try:
        continuation = load_module("k_continuation_clean_under_test", "continuation.py")
    finally:
        sys.path.remove(str(HERE))
    partial = {"status": "failed", "error": {"type": "x"}, "failed_utc": 1.0, "datasets": {}}

    continuation.clean_inherited_failure_fields(partial)

    assert "error" not in partial
    assert "failed_utc" not in partial
