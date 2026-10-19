import importlib
import importlib.util

import numpy as np
import pytest
import torch
from torch import nn


MODULE = "research.incremental_latent_adaptation_20261019_v1.engine"


def _engine():
    assert importlib.util.find_spec(MODULE) is not None, "incremental engine is missing"
    return importlib.import_module(MODULE)


class Correction(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))
        self.dropout = nn.Dropout(.15) if dropout else nn.Identity()

    def forward(self, x, foundation=None):
        return self.dropout(x[:, -1:]) * self.weight


def _data():
    rng = np.random.default_rng(4101)
    x = rng.normal(size=(57, 4, 2)).astype(np.float32)
    g = rng.normal(size=(57, 1, 2)).astype(np.float32) * .1
    y = g + .6 * x[:, -1:]
    mask = np.ones_like(y, dtype=bool)
    mask[::3, :, 1] = False
    return dict(x=x, g=g, y=y, mask=mask, origins=np.arange(100, 157))


def test_search_scores_zero_checkpoint_and_selects_mselect(tmp_path):
    engine = _engine()
    data = _data()
    result = engine.run_search(Correction(), data, np.arange(33), np.arange(36, 57),
                               92601, None, tmp_path, (0, .25, 1, 3), device="cpu")
    assert result["status"] == "COMPLETE"
    assert [row["epoch"] for row in result["curves"]] == [0, .25, 1, 3]
    error = data["g"] - data["y"]
    expected = np.mean([np.square(error[36:, :, c][data["mask"][36:, :, c]]).mean(dtype=np.float64)
                        for c in range(2)])
    assert result["curves"][0]["M_select_mse"] == pytest.approx(expected, rel=1e-6)
    chosen = min(result["curves"], key=lambda row: (row["M_select_mse"], row["epoch"]))
    assert result["selected_epoch"] == chosen["epoch"]
    assert result["exposures"] == 99


def test_extension_restarts_optimizer_sampler_and_rng_exactly(tmp_path):
    engine = _engine()
    data = _data()
    args = (data, np.arange(33), np.arange(36, 57), 92601, None)
    full_model, resumed_model = Correction(True), Correction(True)
    full = engine.run_search(full_model, *args, tmp_path / "full", (0, .25, 1, 3), device="cpu")
    engine.run_search(resumed_model, *args, tmp_path / "resumed", (0, .25, 1), device="cpu")
    torch.manual_seed(800)
    torch.rand(11)
    resumed_model = Correction(True)
    resumed = engine.run_search(resumed_model, *args, tmp_path / "resumed", (0, .25, 1, 3), device="cpu")
    assert torch.equal(full_model.weight, resumed_model.weight)
    assert full["sampled_indices_sha256"] == resumed["sampled_indices_sha256"]
    assert full["updates"] == resumed["updates"]
    for left, right in zip(full["curves"], resumed["curves"]):
        assert left["M_fit_mse"] == right["M_fit_mse"]
        assert left["M_select_mse"] == right["M_select_mse"]


def test_refit_uses_all_mcal_from_fresh_initialization(tmp_path):
    engine = _engine()
    data = _data()
    result = engine.run_refit(Correction(), data, 92601, None, tmp_path, 1, device="cpu")
    assert result["status"] == "COMPLETE"
    assert result["exposures"] == len(data["x"])
    assert result["updates"] == 4
    assert result["fresh_initialization"] is True


def test_resume_rejects_parent_contract_change(tmp_path):
    engine = _engine()
    data = _data()
    args = (data, np.arange(33), np.arange(36, 57), 92601, None, tmp_path, (0, .25))
    engine.run_search(Correction(), *args, contract={"parent_sha": "a"}, device="cpu")
    with pytest.raises(ValueError, match="contract"):
        engine.run_search(Correction(), *args, contract={"parent_sha": "b"}, device="cpu")


def test_cached_readout_never_calls_live_feature_path(tmp_path):
    engine = _engine()
    data = _data()
    data["phi"] = data["x"][:, -1:]

    class CachedReadout(Correction):
        def forward(self, x, foundation=None):
            raise AssertionError("Live features must not run during cached readout fitting")

        def from_features(self, phi):
            return phi * self.weight

    result = engine.run_search(CachedReadout(), data, np.arange(33), np.arange(36, 57),
                               92601, None, tmp_path, (0, .25), device="cpu")
    assert result["status"] == "COMPLETE"
    assert result["training_F_forward_rows"] == 0


def test_deadline_checkpoint_can_resume_before_first_score(tmp_path):
    engine = _engine()
    data = _data()
    args = (data, np.arange(33), np.arange(36, 57), 92601, None, tmp_path, (0, .25))
    partial = engine.run_search(Correction(), *args, deadline=0, device="cpu")
    assert partial["status"] == "PARTIAL_CHECKPOINTED"
    assert partial["curves"] == []
    resumed = engine.run_search(Correction(), *args, device="cpu")
    assert resumed["status"] == "COMPLETE"
    assert resumed["exposures"] == 9


def test_frozen_readout_counts_no_foundation_backward_rows(tmp_path):
    engine = _engine()
    data = _data()

    class Foundation(nn.Module):
        def __init__(self):
            super().__init__()
            self.streams = self.physical_streams = 0

        def forward(self, x):
            self.streams += len(x) * x.shape[-1]
            self.physical_streams += len(x) * x.shape[-1]
            return x

    class Readout(Correction):
        def forward(self, x, foundation=None):
            return foundation(x)[:, -1:] * self.weight

    result = engine.run_search(Readout(), data, np.arange(33), np.arange(36, 57),
                               92601, Foundation(), tmp_path, (0, .25), device="cpu")
    assert result["training_F_forward_rows"] == 18
    assert result["training_F_backward_rows"] == 0


def test_training_only_objective_hook_controls_accumulated_gradient(tmp_path):
    engine = _engine()
    data = _data()

    class PrefixObjective(Correction):
        def __init__(self):
            super().__init__()
            self.training_calls = 0

        def training_loss(self, x, g, target, mask, foundation, counts, n_fit, batch_rows):
            self.training_calls += 1
            prediction = g + self(x, foundation)
            return engine.macro_loss(prediction, 2 * g - target, mask, counts, n_fit, batch_rows)

    model = PrefixObjective()
    result = engine.run_search(model, data, np.arange(33), np.arange(36, 57),
                               92601, None, tmp_path, (0, .25), device="cpu")
    assert model.training_calls == 3
    assert torch.all(model.weight < 0)
    assert result["selected_epoch"] == 0
    assert any(path.endswith("test_incremental_engine.py") for path in result["contract"]["model_sources"])
