import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


@pytest.fixture
def feature_fixture(monkeypatch):
    module = importlib.import_module("research.gt_ts2_fastcheck_20261018_v1.feature_cache")
    values = np.arange(1600, dtype=np.float32)[:, None] / 1000
    physical = np.broadcast_to(values, (1600, 21)).copy()
    data = {"raw_physical": physical, "mask": np.ones_like(physical, bool),
            "mean": np.zeros(21, np.float32), "std": np.ones(21, np.float32),
            "legacy_x": physical.copy(), "train_origins": np.array([1024, 1152]),
            "val_origins": np.array([1400, 1410]), "train_end": 1300, "val_end": 1600,
            "contract_hash": "synthetic_new_fill_contract"}
    network = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        network.weight.fill_(1.)
    encoder = SimpleNamespace(net=network, _net=network, device="cpu")

    def export(model, x):
        representation = x[..., :1].repeat(1, 1, 320)
        representation[..., 0] = 7.
        return representation

    monkeypatch.setattr(module, "load_ts2", lambda *args, **kwargs: encoder)
    monkeypatch.setattr(module, "export_tensor", export)
    monkeypatch.setattr(module, "export_official", lambda model, x: export(model, torch.from_numpy(x)).numpy())
    return module, data, {"seed": 92601, "id": "synthetic_parent", "checkpoint_sha256": "fixture_sha"}


def test_features_are_unstandardized_and_statistics_use_train_only(tmp_path, feature_fixture):
    module, data, binding = feature_fixture
    progress = []
    result = module.build_feature_cache(binding, data, "PRE", 92601, tmp_path / "cache",
                                        callback=lambda *args: progress.append(args), device="cpu")
    with np.load(result["features_paths"]["train"]) as archive:
        features = archive["features"]
        np.testing.assert_array_equal(archive["origins"], [1024, 1152])
    assert features.shape == (2, 256, 320)
    np.testing.assert_array_equal(features[..., 0], 7.)
    assert features[0, 0, 1] == pytest.approx(.0015)
    assert result["mean"][0] == 7. and result["std"][0] == 1.
    assert result["mean"][1] == pytest.approx(.5755)
    assert result["mean"][1] < .7
    expected_std = np.std(features[..., 1].astype(np.float64), ddof=0).astype(np.float32)
    assert result["std"][1] == expected_std
    assert progress
    assert result["receipt"]["encoder_unchanged"]
    assert result["receipt"]["parity"]["cache_online_exact"]
    assert str(tmp_path) not in json.dumps(result["receipt"])


def test_feature_cache_reuses_only_verified_identity(tmp_path, feature_fixture):
    module, data, binding = feature_fixture
    directory = tmp_path / "cache"
    first = module.build_feature_cache(binding, data, "RAND", 92601, directory, device="cpu")
    files_before = {p.name: p.read_bytes() for p in directory.iterdir()}
    second = module.build_feature_cache(binding, data, "RAND", 92601, directory, device="cpu")
    assert first["receipt"]["identity_sha256"] == second["receipt"]["identity_sha256"]
    assert files_before == {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(ValueError, match="identity"):
        module.build_feature_cache(binding, data, "PRE", 92601, directory, device="cpu")


def test_cache_identity_catches_preprocessing_or_origin_changes(tmp_path, feature_fixture):
    module, data, binding = feature_fixture
    directory = tmp_path / "cache"
    module.build_feature_cache(binding, data, "PRE", 92601, directory, device="cpu")
    data["raw_physical"][1023, 1] += 1
    with pytest.raises(ValueError, match="identity"):
        module.build_feature_cache(binding, data, "PRE", 92601, directory, device="cpu")


def test_feature_cache_rejects_corruption_and_partial_attempt(tmp_path, feature_fixture):
    module, data, binding = feature_fixture
    directory = tmp_path / "cache"
    result = module.build_feature_cache(binding, data, "PRE", 92601, directory, device="cpu")
    result["features_paths"]["val"].write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA"):
        module.build_feature_cache(binding, data, "PRE", 92601, directory, device="cpu")
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "train_features.npz").write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        module.build_feature_cache(binding, data, "PRE", 92601, partial, device="cpu")
    assert (partial / "train_features.npz").read_bytes() == b"preserve"


def test_cache_uses_fixed_seed_matching_parent(tmp_path, feature_fixture):
    module, data, binding = feature_fixture
    with pytest.raises(ValueError, match="seed"):
        module.build_feature_cache(binding, data, "PRE", 92602, tmp_path / "cache", device="cpu")
