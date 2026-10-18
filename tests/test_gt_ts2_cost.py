import importlib
from types import SimpleNamespace

import numpy as np
import pytest


def cost():
    try:
        return importlib.import_module("research.gt_ts2_fastcheck_20261018_v1.cost")
    except ModuleNotFoundError:
        pytest.fail("full online inference cost worker is missing")


def test_timing_summary_excludes_warmups_and_uses_whole_batch_quantiles():
    trials = [{"stage": "warmup", "whole_batch_ms": 9999., "batch": 8}]
    trials += [{"stage": "timed", "whole_batch_ms": value, "batch": 8,
                "peak_allocated_MiB": 10., "peak_reserved_MiB": 30., "RSS_MiB": 40.}
               for value in (10., 20., 30., 40.)]
    result = cost().timing_summary(trials)
    assert result["p50_ms"] == 25.
    assert result["q25_ms"] == 17.5
    assert result["q75_ms"] == 32.5
    assert result["iqr_ms"] == 15.
    assert result["p95_ms"] == pytest.approx(38.5)
    assert result["origins_per_second_at_median"] == 320.
    assert result["peak_allocated_MiB"] == 10.
    assert result["peak_reserved_MiB"] == 30.
    assert result["repetitions"] == 4


def test_registered_group_indices_cycle_without_dropping_batch_members():
    contract = {"cost": {"fixed_group_indices": {"8": [0, 181, 363]}}}
    groups = cost().group_indices(contract, 8, 371)
    for iteration in range(20):
        expected = [0, 181, 363][iteration % 3]
        np.testing.assert_array_equal(groups[iteration % 3], np.arange(expected, expected + 8))
    assert groups[-1][-1] == 370
    contract["cost"]["fixed_group_indices"]["8"][-1] = 364
    with pytest.raises(ValueError, match="group"):
        cost().group_indices(contract, 8, 371)


def test_timed_trial_keeps_resident_memory_and_counts_real_foundation_calls(monkeypatch):
    module = cost()
    clock = SimpleNamespace(now=0., resets=0, syncs=0, peak=64 * 1024**2)
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda: setattr(clock, "syncs", clock.syncs + 1))

    def reset():
        clock.resets += 1
        clock.peak = 64 * 1024**2

    monkeypatch.setattr(module.torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(module.torch.cuda, "max_memory_allocated", lambda: clock.peak)
    monkeypatch.setattr(module.torch.cuda, "max_memory_reserved", lambda: 96 * 1024**2)
    foundation = SimpleNamespace(calls=7, streams=99, context_records=[])
    ctx = {"data": {"mean": np.full(21, 3., np.float32), "std": np.full(21, 2., np.float32)}}
    group = {"indices": np.array([0, 1]), "origins": np.array([1024, 1048]),
             "physical": np.ones((2, 16, 21), np.float32), "mask": np.ones((2, 16, 21), bool)}

    def online(*args):
        clock.now += .0125
        clock.peak += 8 * 1024**2
        foundation.calls += 2
        foundation.streams += 84
        foundation.context_records.append({"requested_context_length": 4, "actual_context_length": 4,
                                           "streams": 84, "row_sizes": [84]})
        return np.full((2, 48, 21), 5., np.float64)

    monkeypatch.setattr(module, "_online_physical", online)
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=120 * 1024**2))
    row = module._trial("timed", 0, group, ctx, {}, foundation,
                        np.ones((2, 48, 21), np.float32), process)
    assert row["whole_batch_ms"] == 12.5
    assert row["peak_allocated_MiB"] == 72.
    assert row["peak_reserved_MiB"] == 96.
    assert row["RSS_MiB"] == 120.
    assert row["F_calls"] == 2 and row["F_streams"] == 84
    assert row["origins_per_second"] == 160.
    assert row["prediction_parity"]["status"] == "PASS"
    assert row["TS2cache_used"] is False
    assert clock.resets == 1 and clock.syncs == 2


def test_warmup_trial_does_not_reset_cumulative_warmup_peak(monkeypatch):
    module = cost()
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(module.torch.cuda, "reset_peak_memory_stats", lambda: pytest.fail("warmup peak reset inside repetition"))
    monkeypatch.setattr(module.torch.cuda, "max_memory_allocated", lambda: 80 * 1024**2)
    monkeypatch.setattr(module.torch.cuda, "max_memory_reserved", lambda: 100 * 1024**2)
    monkeypatch.setattr(module, "_online_physical", lambda *args: np.ones((1, 48, 21), np.float64))
    ctx = {"data": {"mean": np.zeros(21), "std": np.ones(21)}}
    group = {"indices": np.array([0]), "origins": np.array([1024]),
             "physical": np.ones((1, 16, 21)), "mask": np.ones((1, 16, 21), bool)}
    foundation = SimpleNamespace(calls=0, streams=0, context_records=[])
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1))
    row = module._trial("warmup", 5, group, ctx, {}, foundation, np.ones((1, 48, 21)), process)
    assert row["memory_peak_scope"] == "cumulative warmup phase"
    assert row["peak_allocated_MiB"] == 80.


def test_timing_summary_rejects_empty_or_zero_duration_measurements():
    with pytest.raises(ValueError):
        cost().timing_summary([])
    with pytest.raises(ValueError):
        cost().timing_summary([{"stage": "timed", "whole_batch_ms": 0., "batch": 1}])


def test_resident_inventory_deduplicates_aliases_and_uses_actual_dtype_bytes():
    module = cost()
    first = module.torch.nn.Linear(2, 3, dtype=module.torch.float64)
    first.register_buffer("stats", module.torch.tensor([1., 2.], dtype=module.torch.float32))
    second = module.torch.nn.Linear(3, 1, bias=False, dtype=module.torch.float16)
    second.register_buffer("mask", module.torch.ones(4, dtype=module.torch.bool))
    inventory = module.resident_inventory({"primary": first, "alias": first, "separate": second})
    assert inventory["by_device"]["cpu"]["parameters"] == 12
    assert inventory["by_device"]["cpu"]["parameter_bytes"] == 78
    assert inventory["by_device"]["cpu"]["buffer_bytes"] == 12
    assert inventory["by_device"]["cuda"]["parameters"] == 0
    assert inventory["by_module"]["alias"]["cpu"]["parameters"] == 0
    assert inventory["total_unique_parameters"] == 12
    assert inventory["total_unique_parameter_bytes"] == 78
    assert inventory["total_unique_buffer_bytes"] == 12


def test_online_boundary_fills_locally_and_returns_physical_fp64(monkeypatch):
    module = cost()
    execution = importlib.import_module("research.gt_ts2_fastcheck_20261018_v1.execution")
    original_tensor = module.torch.as_tensor
    observed = []

    def cpu_transfer(value, *, dtype, device):
        assert device == "cuda"
        return original_tensor(value, dtype=dtype)

    def forward(route, foundation, x):
        observed.append(x.detach().numpy().copy())
        return x[:, -1:].repeat(1, 48, 1)

    monkeypatch.setattr(module.torch, "as_tensor", cpu_transfer)
    monkeypatch.setattr(execution, "route_forward", forward)
    physical = np.full((1, 2, 21), 5., np.float32)
    physical[:, 0] = np.nan
    mask = np.ones_like(physical, bool)
    mask[:, 0] = False
    ctx = {"data": {"mean": np.full(21, 3., np.float32), "std": np.full(21, 2., np.float32)}}
    forecast = module._online_physical(ctx, {}, None, physical, mask)
    np.testing.assert_array_equal(observed[0][:, 0], 0.)
    np.testing.assert_array_equal(observed[0][:, 1], 1.)
    np.testing.assert_array_equal(forecast, 5.)
    assert forecast.dtype == np.float64


@pytest.mark.parametrize("run_id,expected_updates", [
    ("A_RAW_L512", 0), ("A_GT_L512_s92601", 0),
    ("B_GT_FT_L512_s92601", 100), ("B_GT_FT_L512_s92601", 300),
])
def test_measure_loads_once_and_seals_all_thirty_cyclic_trials(tmp_path, monkeypatch, run_id, expected_updates):
    module = cost()
    execution = importlib.import_module("research.gt_ts2_fastcheck_20261018_v1.execution")
    monkeypatch.setattr(module.c, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(module.c, "REPORT", tmp_path / "report")
    module.c.CACHE.mkdir(); module.c.REPORT.mkdir()
    contract = {"cost": {"warmups": 10, "repetitions": 20, "groups_cycle": True,
                         "full_online_TS2": True, "fixed_group_indices": {"1": [0, 1, 2]}}}
    module.c.seal_json(module.c.REPORT / "EXPERIMENT_CONTRACT.json", contract)
    if expected_updates:
        module.c.seal_json(module.c.REPORT / "UPDATE_DECISION.json", {"updates": expected_updates})
        checkpoint = module.c.CACHE / "training" / run_id / f"checkpoint_{expected_updates:06d}.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"CPU fixture selected checkpoint")
    prediction = np.ones((3, 48, 21), np.float32)
    origins = np.array([1024, 1048, 1072])
    data = {"val_origins": origins, "mean": np.zeros(21, np.float32), "std": np.ones(21, np.float32),
            "raw_physical": np.ones((1200, 21), np.float32), "mask": np.ones((1200, 21), bool),
            "val_end": 1200, "contract_hash": "fixture_contract"}
    archive = module.c.CACHE / "predictions" / f"{run_id}.npz"
    module.c.seal_npz(archive, prediction=prediction, origins=origins, mean=data["mean"], std=data["std"])
    module.c.seal_json(module.c.REPORT / "receipts" / f"{run_id}.json",
                       {"status": "COMPLETE", "run_id": run_id, "data_contract_sha256": "fixture_contract",
                        "artifact_sha256": module.c.file_sha(archive), "prediction_sha256": module.c.array_sha(prediction)})
    network = module.torch.nn.Linear(1, 1)
    foundation = SimpleNamespace(model=network, row_microbatch=128, calls=0, streams=0, context_records=[],
                                 verify_unchanged=lambda: {"unchanged": True, "parameter_gradients_absent": True})
    loads = []
    ctx = {"data": data, "parents": {"92601": {"GT": {"checkpoint_sha256": "fixture_sha", "source_pins": {}}}}}
    monkeypatch.setattr(execution, "context", lambda: ctx)
    registered_specs = execution.candidate_specs()
    monkeypatch.setattr(execution, "candidate_specs", lambda: registered_specs)
    monkeypatch.setattr(execution, "cuda_setup", lambda: None)
    monkeypatch.setattr(execution, "foundation", lambda ctx: loads.append("F") or foundation)
    monkeypatch.setattr(execution, "build_route", lambda spec, ctx, f, **kwargs: loads.append("route") or {"spec": spec})
    monkeypatch.setattr(module.c, "progress", lambda *args, **kwargs: None)
    resets = []
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(module.torch.cuda, "reset_peak_memory_stats", lambda: resets.append(1))
    monkeypatch.setattr(module.torch.cuda, "get_device_name", lambda: "CPU fixture GPU boundary")
    for name, value in (("memory_allocated", 64), ("memory_reserved", 96),
                        ("max_memory_allocated", 72), ("max_memory_reserved", 100)):
        monkeypatch.setattr(module.torch.cuda, name, lambda value=value: value * 1024**2)

    def online(ctx, route, f, physical, mask):
        f.calls += 1
        f.streams += 21
        f.context_records.append({"requested_context_length": 512, "actual_context_length": 512,
                                  "streams": 21, "row_sizes": [21]})
        return np.ones((1, 48, 21), np.float64)

    monkeypatch.setattr(module, "_online_physical", online)
    receipt = module.measure(run_id, 1)
    assert receipt["spec"]["updates"] == expected_updates
    assert receipt["summary"]["updates"] == expected_updates
    assert all(row["updates"] == expected_updates for row in receipt["trials"])
    original = next(spec for spec in registered_specs if spec["run_id"] == run_id)
    assert original["updates"] is None if expected_updates else original["updates"] == 0
    assert loads == ["F", "route"]
    assert len(resets) == 21
    assert len(receipt["trials"]) == 30
    assert [row["groupstart"] for row in receipt["trials"][:10]] == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    assert [row["groupstart"] for row in receipt["trials"][10:]] == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]
    assert receipt["summary"]["repetitions"] == 20
    assert receipt["validation"]["status"] == "PASS"
    assert receipt["validation"]["prediction_parity_trials"] == 30
    assert receipt["loaded_baseline"]["allocated_MiB"] == 64.
    assert receipt["summary"]["peak_allocated_MiB"] == 72.
    assert receipt["resident_inventory"]["by_device"]["cpu"]["parameters"] == 2
    assert receipt["resident_inventory"]["by_device"]["cpu"]["parameter_bytes"] == 8
    assert receipt["summary"]["resident_parameters"] == 0
    assert receipt["summary"]["CPU_resident_parameters"] == 2
    output = module.c.REPORT / "cost_receipts" / f"{run_id}_B1.json"
    assert output.is_file()
    assert str(tmp_path) not in output.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.measure(run_id, 1)
