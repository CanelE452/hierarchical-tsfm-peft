"""Operational cost deadline checks on synthetic CPU routes only."""
from types import SimpleNamespace

import numpy as np
import pytest

from research.raw_context_multiscale_20261018_v1 import costs


def synthetic_task():
    values = np.arange(40, dtype=np.float32)[:, None]
    return dict(L=4, H=2, columns=["c0"], physical=values,
                observed=np.ones(values.shape, bool), split_ranges=dict(V_select=(8, 24)))


class CpuRoute:
    def __init__(self, task, clock, family="CURRENT", load_seconds=0.):
        self.task, self.clock, self.family = task, clock, family
        self.calls, self.loads, self.activated, self.closed = 0, 0, False, False
        self.mean, self.std = np.zeros(1, np.float32), np.ones(1, np.float32)
        self.spec, self.model, self.g_route, self.b_route, self.ridge_artifact = {}, None, None, None, None
        self.pool = SimpleNamespace(foundation=None, get=self.load, instances_created=0)
        self.load_seconds = load_seconds
        self.last_receipt = dict(F_logical_streams=0, F_physical_streams=0, F_calls=0, F_invocations=[])
    def load(self, horizon):
        self.loads += 1
        self.clock[0] += self.load_seconds
    def activate(self):
        self.activated = True
    def predict_physical(self, physical, observed):
        self.calls += 1
        self.clock[0] += 1.
        return np.repeat(physical[:, -1:], self.task["H"], axis=1)
    def close(self):
        self.closed = True


@pytest.mark.parametrize("n_origins", [3, 5])
def test_full_parity_checks_deadline_before_each_batch_and_after_last_batch(monkeypatch, n_origins):
    task, clock = synthetic_task(), [0.]
    route = CpuRoute(task, clock)
    origins = costs.fixed_origins(task)[:n_origins]
    inputs = costs.windows(task, origins)
    monkeypatch.setattr(costs.time, "time", lambda: clock[0])
    with pytest.raises(TimeoutError, match="budget reached"):
        costs.whole_v_parity(route, task, origins, inputs, None, 1,
                             deadline_request=dict(_deadline_unix=2.5))
    assert route.calls == 3


def test_expired_route_load_cannot_start_foundation_load(monkeypatch):
    task, clock = synthetic_task(), [0.]
    route = CpuRoute(task, clock, family="RAW")
    request = dict(dataset="jena", accuracy_run_id="synthetic", cost_id="synthetic_B1", batch_size=1,
                   device="cpu", route_spec=dict(family="RAW"), _deadline_unix=4.)
    monkeypatch.setattr(costs.time, "time", lambda: clock[0])
    def loader():
        clock[0] = 5.
        return route
    with pytest.raises(TimeoutError, match="budget reached"):
        costs.evaluate_inputs(request, {}, task, loader)
    assert route.loads == route.calls == 0
    assert route.closed is True


def test_expired_foundation_load_cannot_start_activation_or_parity(monkeypatch):
    task, clock = synthetic_task(), [0.]
    route = CpuRoute(task, clock, family="RAW", load_seconds=5.)
    request = dict(dataset="jena", accuracy_run_id="synthetic", cost_id="synthetic_B1", batch_size=1,
                   device="cpu", route_spec=dict(family="RAW"), _deadline_unix=4.)
    monkeypatch.setattr(costs.time, "time", lambda: clock[0])
    with pytest.raises(TimeoutError, match="budget reached"):
        costs.evaluate_inputs(request, {}, task, lambda: route)
    assert route.loads == 1
    assert route.calls == 0 and route.activated is False
    assert route.closed is True
