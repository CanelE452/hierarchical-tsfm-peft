import importlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from research.raw_context_multiscale_20261018_v1 import common, gates


@pytest.fixture
def ctx(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "research/informs_broad_validation_20261015_v1"))
    modules = {key: importlib.import_module(name) for key, name in
               (("fit", "fit_base"), ("training", "training"), ("data", "data_contract"))}
    torch.set_num_threads(2)
    tasks = {}
    for dataset, config in common.EXPECTED.items():
        values = np.random.default_rng(79).normal(size=(576, config["C"])).astype(np.float32)
        observed = np.ones_like(values, bool)
        observed[::11, 1] = False
        tasks[dataset] = dict(dataset=dataset, K=config["K"], L=512, H=config["H"],
            physical=values, observed=observed, columns=[str(index) for index in range(config["C"])],
            data_sha256=modules["data"].array_sha256(values),
            split_ranges=dict(A_fit=(0, 576), A_stop=(576, 640), M_cal=(640, 704),
                              V_select=(704, 768), T_final=(768, 832)))
    return dict(repo=root, cache=tmp_path / "cache", report=tmp_path / "report", asset=tmp_path,
                modules=modules, tasks=tasks, device="cpu")


class FrozenFixture(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.7), requires_grad=False)
        self.horizon = horizon
        self.row_microbatch = 128
        self.initial_hash = "fixture-frozen-F"
        self.model = nn.Identity()
        self.model.chronos_config = SimpleNamespace(prediction_length=64)
        self.streams = self.row_calls = self.physical_streams = self.physical_calls = 0

    def forward(self, z):
        self.streams += len(z) * z.shape[-1]
        self.physical_streams += len(z) * z.shape[-1]
        self.row_calls += 1
        self.physical_calls += 1
        return (z[:, -1:] * self.weight + .2 * z.mean(1, keepdim=True)).expand(-1, self.horizon, -1)

    def assert_frozen_unchanged(self):
        assert self.weight.grad is None and self.weight.item() == torch.tensor(.7).item()


def test_train_only_cpu_gate_schema_gradients_and_no_cuda_allocation(ctx, monkeypatch):
    loads = []

    def load(local, modules, horizon, device):
        assert local["device"] == "cpu" and device == "cpu"
        loads.append(horizon)
        return FrozenFixture(horizon)

    def forbidden(*args, **kwargs):
        pytest.fail("CPU gate called a CUDA allocation/synchronization operation")

    monkeypatch.setattr(gates, "load_foundation", load)
    for name in ("synchronize", "reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    value = gates.run_gates(ctx)
    assert value["status"] == "PASS", value.get("reason")
    assert loads == [48, 12]
    assert value["new_A_stop_V_T_scores"] == 0 and value["GPU_seconds"] == 0
    assert value["profiles"] == [] and value["CUDA_requested"] is False
    assert value["discarded_preparation_and_profile_seconds"] > 0
    assert len(value["datasets"]) == 2
    for record in value["datasets"]:
        assert record["first_hidden_gradient_zero_is_legitimate"]
        assert record["training_gradient_path"]["shared_instance_count"] == 1
        assert record["training_gradient_path"]["changed"] == {"front.": True, "E": True, "D": True}
        calls = record["training_gradient_path"]["calls"]
        assert calls[0]["input_shape"][1] == 512
        assert calls[1]["input_shape"][1] == 256
        assert record["actual_fixed_F_head_horizon"] == 64
        assert len(record["CP0"]) == 2
        assert record["frozen_F_weights_unchanged"] is True
    saved = common.read(Path(ctx["report"]) / "VALIDATION_SMOKE_CPU.json")
    assert saved["status"] == "PASS"
    assert not list(Path(ctx["cache"]).rglob("*.npz"))
    with pytest.raises(FileExistsError, match="sealed gate"):
        gates.run_gates(ctx)


def test_failure_is_recorded_without_silent_tolerance_change(ctx, monkeypatch):
    monkeypatch.setattr(gates, "load_foundation", lambda local, modules, horizon, device: FrozenFixture(horizon))

    def fail(*args, **kwargs):
        raise AssertionError("deliberate exact parity failure")

    monkeypatch.setattr(gates, "_dataset_gates", fail)
    value = gates.run_gates(ctx, output_id="failed_fixture")
    assert value["status"] == "FAIL" and value["reason"] == "deliberate exact parity failure"
    assert value["tolerances"] == dict(atol=1e-5, rtol=1e-4)
    assert common.read(Path(ctx["report"]) / "VALIDATION_SMOKE_CPU_failed_fixture.json")["status"] == "FAIL"


def test_profile_and_output_identifiers_validate_before_foundation_load(ctx):
    with pytest.raises(ValueError, match="requires CUDA"):
        gates.run_gates(ctx, profile=True)
    with pytest.raises(ValueError, match="safe component"):
        gates.run_gates(ctx, output_id="../escape")
    with pytest.raises(ValueError, match="cpu or cuda"):
        gates.run_gates(ctx, device="other")


def test_macro_profile_update_matches_one_effective_batch():
    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(.7))

        def forward(self, x, foundation):
            return x * self.weight

    torch.manual_seed(1)
    model, reference = Adapter(), Adapter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
    ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=.001, weight_decay=0)
    model._profile_optimizer = optimizer
    x, y = torch.randn(16, 3, 4), torch.randn(16, 3, 4)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[::2, :, 0] = False
    gates._profile_step(model, None, x, y, mask)
    ref_optimizer.zero_grad(set_to_none=True)
    gates._loss(reference(x, None), y, mask).backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1)
    ref_optimizer.step()
    torch.testing.assert_close(model.weight, reference.weight, atol=0, rtol=0)


def test_expired_gate_records_failure_before_any_data_or_foundation_load(ctx, monkeypatch):
    ctx.pop("tasks")
    ctx["deadline"] = 4.
    monkeypatch.setattr(gates, "time", SimpleNamespace(time=lambda: 5., perf_counter=lambda: 5.))
    monkeypatch.setattr(common, "load_task", lambda *args: pytest.fail("Expired gate loaded data"))
    monkeypatch.setattr(gates, "load_foundation", lambda *args, **kwargs: pytest.fail("Expired gate loaded F"))
    value = gates.run_gates(ctx, output_id="expired")
    assert value["status"] == "FAIL" and value["error_type"] == "TimeoutError"
    assert value["datasets"] == value["profiles"] == []


def test_gate_checks_deadline_after_foundation_load_before_dataset_work(ctx, monkeypatch):
    clock, loads = [0.], []
    ctx["deadline"] = 4.
    monkeypatch.setattr(gates, "time", SimpleNamespace(time=lambda: clock[0], perf_counter=lambda: clock[0]))
    def load(local, modules, horizon, device):
        loads.append(horizon)
        clock[0] = 5.
        return FrozenFixture(horizon)
    monkeypatch.setattr(gates, "load_foundation", load)
    monkeypatch.setattr(gates, "_dataset_gates", lambda *args: pytest.fail("Expired F load entered dataset gates"))
    value = gates.run_gates(ctx, output_id="expired_after_load")
    assert value["status"] == "FAIL" and value["error_type"] == "TimeoutError"
    assert loads == [48]
    assert value["foundation_load_seconds"] == 5.


def test_dataset_gate_checks_deadline_between_model_initializations(ctx, monkeypatch):
    clock, created = [0.], []
    ctx["deadline"] = 4.
    monkeypatch.setattr(gates, "time", SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(gates.training, "_factory_context", lambda value: value)
    @contextmanager
    def factory(local):
        def create(spec, *args):
            created.append(spec["scientific_identity"]["raw_context_mode"])
            clock[0] = 5.
            return None, None
        yield create
    monkeypatch.setattr(gates.models, "model_factory", factory)
    with pytest.raises(TimeoutError, match="budget reached"):
        gates._dataset_gates(ctx, ctx["tasks"]["jena"], None, "cpu")
    assert created == ["S"]


def test_profile_step_stops_before_expired_next_microbatch(monkeypatch):
    clock, calls = [0.], [0]
    monkeypatch.setattr(gates, "time", SimpleNamespace(time=lambda: clock[0]))
    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(.7))
        def forward(self, x, foundation):
            calls[0] += 1
            clock[0] = 5.
            return x * self.weight
    model = Adapter()
    model._profile_optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    x = torch.ones(16, 2, 1)
    with pytest.raises(TimeoutError, match="budget reached"):
        gates._profile_step(model, None, x, x, torch.ones_like(x, dtype=torch.bool), deadline=4.)
    assert calls[0] == 1
    assert model.weight.item() == torch.tensor(.7).item()
