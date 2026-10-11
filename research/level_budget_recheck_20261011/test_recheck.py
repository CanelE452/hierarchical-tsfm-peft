from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark
import run


def test_select_milestone_bests_uses_earliest_tie_within_each_window():
    # Break caught: milestone selection reuses the legacy early-stop epoch or takes the last tie.
    curves = [
        {"epoch": 0, "val_reconstruction": {"mse": 5.0}},
        {"epoch": 2, "val_reconstruction": {"mse": 3.0}},
        {"epoch": 4, "val_reconstruction": {"mse": 3.0}},
        {"epoch": 7, "val_reconstruction": {"mse": 2.0}},
    ]

    selected = run.select_milestone_bests(curves, [4, 7])

    assert selected[4]["epoch"] == 2
    assert selected[7]["epoch"] == 7


def test_protocol_adapter_and_benchmark_case_counts_match_budget_contract():
    # Break caught: PCA is duplicated by milestone or RAW_F0 is accidentally passed through ridge.
    protocol = run.read_json(Path("research/level_budget_recheck_20261011/protocol.json"))

    assert len(run.decoder_adapter_ids(protocol)) == 9
    assert len(benchmark.case_rows({"protocol": protocol, "datasets": {d: {"benchmark_cases": run.benchmark_case_ids(protocol)} for d in protocol["datasets"]}})) == 24
    assert run.benchmark_case_ids(protocol).count("PCA_ANCHOR") == 1
    assert run.benchmark_case_ids(protocol).count("RAW_F0") == 1


def test_epoch_update_contract_is_fixed_no_earlystop():
    # Break caught: implementation silently retains parent early stopping or changes per-epoch updates.
    protocol = run.read_json(Path("research/level_budget_recheck_20261011/protocol.json"))

    assert protocol["early_stopping"] is False
    assert protocol["updates_per_epoch"] == protocol["epoch_origins"] // protocol["batch"]
    assert protocol["total_updates_per_fit"] == protocol["max_epochs"] * protocol["updates_per_epoch"]
    assert protocol["additional_updates_per_fit"] == protocol["additional_epochs"] * protocol["updates_per_epoch"]


def test_raw_equivalent_normalization_roundtrip_keeps_missing_zero_contract():
    # Break caught: benchmark reports standardization but only passes pre-standardized inputs through unchanged.
    standardized = np.array([[[0.0, 2.0], [-1.0, 0.0]]], dtype=np.float32)
    mean = np.array([10.0, -3.0], dtype=np.float32)
    std = np.array([2.0, 4.0], dtype=np.float32)

    raw = benchmark.raw_from_standardized(standardized, mean, std)
    normalized = benchmark.normalize_raw(raw, mean, std)

    assert raw.dtype == np.float64
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized, standardized, rtol=0, atol=1e-6)
    np.testing.assert_allclose(raw[0, 0], mean + standardized[0, 0] * std)
    assert raw[0, 0, 0] == pytest.approx(mean[0])


def test_pca_tensor_encode_reuses_resident_basis(monkeypatch):
    # Break caught: PCA basis is copied to the GPU on every timed call unlike learned resident weights.
    import torch

    calls = {"to": 0}
    original_to = torch.Tensor.to

    def counting_to(self, *args, **kwargs):
        calls["to"] += 1
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", counting_to)
    context = np.ones((2, 3, 2), dtype=np.float32)
    basis = torch.eye(2, 1)

    encoded = benchmark.encode_for_benchmark(context, basis, "cpu")

    np.testing.assert_allclose(encoded, np.ones((2, 3, 1), dtype=np.float32))
    assert calls["to"] == 1


def test_timing_indices_wrap_when_robin_val_count_is_smaller_than_batch32():
    # Break caught: B32 benchmark fails or silently changes batch size when VAL has only 30 origins.
    indices = benchmark.make_timing_indices(total=30, batch=32)

    assert indices.shape == (32,)
    assert indices[:30].tolist() == list(range(30))
    assert indices[30:].tolist() == [0, 1]


def test_case_rows_rotate_learned_cases_between_datasets():
    # Break caught: benchmark always runs cases in one fixed family order.
    protocol = run.read_json(Path("research/level_budget_recheck_20261011/protocol.json"))
    cases = run.benchmark_case_ids(protocol)
    fake = {"protocol": protocol, "datasets": {d: {"benchmark_cases": cases} for d in protocol["datasets"]}}

    rows = benchmark.case_rows(fake)
    robin_learned = [r["adapter"] for r in rows if r["dataset"] == "robin" and r["batch"] == 1 and r["adapter"] not in ("PCA_ANCHOR", "RAW_F0")]
    jena_learned = [r["adapter"] for r in rows if r["dataset"] == "jena" and r["batch"] == 1 and r["adapter"] not in ("PCA_ANCHOR", "RAW_F0")]

    assert jena_learned[0] == robin_learned[1]


def test_atomic_json_refuses_writes_outside_new_campaign_folders(tmp_path):
    # Break caught: new run accidentally writes into parent result/cache folders.
    outside = tmp_path / "outside.json"

    with pytest.raises(PermissionError):
        run.atomic_json(outside, {"bad": True})
