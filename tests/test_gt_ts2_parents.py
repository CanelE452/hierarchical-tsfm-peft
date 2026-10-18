import hashlib
import importlib
import json
import random

import numpy as np
import pytest
import torch

from research.professor_compressor_direct_20261011.models_gt import GTConvAE
from research.professor_compressor_direct_20261011.ts2_adapter import (
    ReconstructionDecoder, create_ts2,
)


def parents():
    return importlib.import_module("research.gt_ts2_fastcheck_20261018_v1.parents")


def binding(path, seed=92601):
    return {"checkpoint": path, "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seed": seed, "id": f"fixture_s{seed}"}


def snapshot():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def assert_rng_equal(before):
    assert random.getstate() == before[0]
    current = np.random.get_state()
    assert current[0] == before[1][0]
    np.testing.assert_array_equal(current[1], before[1][1])
    assert current[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])


@pytest.fixture
def ts2_binding(tmp_path):
    model = create_ts2(21, 320, 92601, device="cpu", batch_size=4, max_train_length=3000)
    with torch.no_grad():
        model.net.module.input_fc.weight.add_(0.25)
    path = tmp_path / "encoder_swa.pt"
    torch.save(model.net.state_dict(), path)
    return binding(path), model.net.state_dict()


def test_gt_loads_saved_graph_and_preserves_rng(tmp_path):
    model = GTConvAE(torch.tensor([[0., .5], [.5, 0.]]), no_spatial=False)
    path = tmp_path / "gt.pt"
    torch.save({"state_dict": model.state_dict()}, path)
    before = snapshot()
    loaded = parents().load_gt(binding(path))
    assert_rng_equal(before)
    assert not loaded.no_spatial
    assert torch.equal(loaded.product.graph, model.product.graph)
    x = torch.arange(32, dtype=torch.float32).reshape(1, 16, 2) / 10
    torch.testing.assert_close(loaded(x), model(x), rtol=0, atol=0)


def test_ts2_loads_swa_and_freezes_both_networks_without_rng_drift(ts2_binding):
    item, expected = ts2_binding
    before = snapshot()
    model = parents().load_ts2(item)
    assert_rng_equal(before)
    for name, tensor in expected.items():
        assert torch.equal(model.net.state_dict()[name], tensor)
    assert not torch.equal(model.net.module.input_fc.weight, model._net.input_fc.weight)
    assert all(not p.requires_grad for n in (model.net, model._net) for p in n.parameters())
    assert not model.net.training and not model._net.training
    assert next(model._net.parameters()).device.type == "cpu"


def test_rand_ts2_is_seed_repeatable_and_does_not_consume_global_rng():
    before = snapshot()
    a = parents().load_ts2({"seed": 92601}, random=True)
    assert_rng_equal(before)
    random.random(); np.random.random(); torch.rand(2)
    b = parents().load_ts2({"seed": 92601}, random=True)
    c = parents().load_ts2({"seed": 92602}, random=True)
    assert torch.equal(a.net.module.input_fc.weight, b.net.module.input_fc.weight)
    assert not torch.equal(a.net.module.input_fc.weight, c.net.module.input_fc.weight)


def test_tensor_export_matches_official_full_window_and_preserves_input(ts2_binding):
    model = parents().load_ts2(ts2_binding[0])
    x = torch.arange(2 * 16 * 21, dtype=torch.float32).reshape(2, 16, 21) / 100
    x[0, 2, 3] = float("nan")
    original = x.clone()
    model.net.train(); model._net.train()
    before = snapshot()
    result = parents().export_tensor(model, x.requires_grad_())
    assert_rng_equal(before)
    reference = parents().export_official(model, original.numpy())
    assert_rng_equal(before)
    assert result.shape == (2, 16, 320)
    assert not result.requires_grad
    np.testing.assert_allclose(result.numpy(), reference, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(x, original, equal_nan=True)
    assert not model.net.training and not model._net.training


def test_tensor_export_uses_swa_not_online(ts2_binding):
    model = parents().load_ts2(ts2_binding[0])
    x = torch.ones(1, 8, 21)
    expected = parents().export_official(model, x.numpy())
    with torch.no_grad():
        model._net.input_fc.weight.fill_(100.)
    np.testing.assert_allclose(parents().export_tensor(model, x).numpy(), expected, rtol=1e-6, atol=1e-6)


def test_decoder_load_preserves_rng_and_saved_forecast_state(tmp_path):
    model = ReconstructionDecoder(320, 21)
    path = tmp_path / "decoder.pt"
    torch.save({"state_dict": model.state_dict()}, path)
    item = {"decoder_checkpoint": path,
            "decoder_checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    before = snapshot()
    loaded = parents().load_ts2_decoder(item)
    assert_rng_equal(before)
    x = torch.ones(1, 3, 320)
    torch.testing.assert_close(loaded(x), model(x), rtol=0, atol=0)


def test_loader_rejects_changed_checkpoint_before_deserialization(tmp_path):
    path = tmp_path / "gt.pt"
    path.write_bytes(b"untrusted changed bytes")
    with pytest.raises(ValueError, match="SHA"):
        parents().load_gt({"checkpoint": path, "checkpoint_sha256": "0" * 64})


@pytest.fixture
def parent_tree(tmp_path, monkeypatch):
    module = parents()
    source_root, asset_root = tmp_path / "source", tmp_path / "assets"
    base = source_root / "research/professor_compressor_direct_20261011"
    base.mkdir(parents=True)
    source = base / "fixture_source.py"
    source.write_text("pass\n")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    snapshot_path = base / "source_snapshots/protocol_memory_retry_v3/protocol_v2.json"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_text('{"version": 2}')
    snapshot_sha = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    jobs, run_rows, result_pins, gt_pins = [], [], {}, {}
    for seed in (92601, 92602):
        for family in ("GT", "TS2"):
            identifier = f"jena_GT_full_s{seed}" if family == "GT" else f"jena_TS2_d320_s{seed}_retry_batch4_v4"
            job = {"id": identifier, "seed": seed, "dataset": "jena", "family": family}
            job.update({"no_spatial": False} if family == "GT" else {"dim": 320})
            jobs.append(job)
            run_rows.append({"job": job, "status": "complete_saved_predictions_verified",
                             "result_path": f"research/professor_compressor_direct_20261011/runs/{identifier}/result.json"})
            cache = asset_root / ".cache/professor_compressor_direct_20261011/runs" / identifier
            cache.mkdir(parents=True)
            prediction = cache / "prediction.npz"
            np.savez(prediction, targets=np.zeros((371, 48, 21), np.float32),
                     CALIBRATED_F=np.ones((371, 48, 21), np.float32),
                     mask=np.ones((371, 48, 21), bool), origins=np.arange(371) + 512,
                     mean=np.zeros(21, np.float32), std=np.ones(21, np.float32))
            diagnostic = cache / "diagnostic.npz"
            np.savez(diagnostic, past=np.zeros((3, 512, 21), np.float32), indices=np.arange(3))
            checkpoint = cache / ("forecast_calibrated_best.pt" if family == "GT" else "encoder_swa.pt")
            checkpoint.write_bytes(identifier.encode())
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            relative = lambda path: path.relative_to(asset_root).as_posix()
            source_pins = {source.relative_to(source_root).as_posix(): source_sha}
            if family == "GT":
                source_pins["research/professor_compressor_direct_20261011/protocol.json"] = snapshot_sha
                gt_pins[identifier] = checkpoint_sha
                training = {"selected_epoch": 1, "selection": "development VAL reconstruction"}
                calibration = {"checkpoint_path": relative(checkpoint), "selected_epoch": 2,
                               "selection": "development VAL forecast"}
            else:
                decoder = cache / "forecast_calibrated_decoder_best.pt"
                decoder.write_bytes(b"decoder" + identifier.encode())
                training = {"encoder": {"checkpoint": relative(checkpoint), "checkpoint_sha256": checkpoint_sha,
                            "configuration": {"input_dims": 21, "output_dims": 320, "hidden_dims": 64,
                            "depth": 10, "batch_size": 4, "max_train_length": 3000}},
                            "decoder_reconstruction": {"best_epoch": 3, "checkpoint_selection": "development VAL reconstruction"}}
                calibration = {"best_checkpoint": relative(decoder),
                               "best_checkpoint_sha256": hashlib.sha256(decoder.read_bytes()).hexdigest(),
                               "best_epoch": 4, "checkpoint_selection": "development VAL forecast"}
            result = {"status": "complete", "job": job, "source_pins": source_pins,
                      "data_contract_hash": "fixture_contract", "training": training, "calibration": calibration,
                      "prediction_path": relative(prediction), "diagnostic_path": relative(diagnostic),
                      "prediction_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
                      "diagnostic_sha256": hashlib.sha256(diagnostic.read_bytes()).hexdigest()}
            result_path = base / "runs" / identifier / "result.json"
            result_path.parent.mkdir(parents=True)
            result_path.write_text(json.dumps(result))
            result_pins[identifier] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    (base / "protocol.json").write_text(json.dumps({"jobs": jobs}))
    (base / "runs.json").write_text(json.dumps({"runs": run_rows}))
    controls = {"protocol.json": hashlib.sha256((base / "protocol.json").read_bytes()).hexdigest(),
                "runs.json": hashlib.sha256((base / "runs.json").read_bytes()).hexdigest()}
    monkeypatch.setattr(module, "ROOT", source_root)
    monkeypatch.setattr(module, "RESULT_PINS", result_pins, raising=False)
    monkeypatch.setattr(module, "GT_CHECKPOINT_PINS", gt_pins, raising=False)
    monkeypatch.setattr(module, "CONTROL_PINS", controls, raising=False)
    return module, asset_root, base


def test_bindings_keep_arrays_private_and_validate_historical_protocol(parent_tree):
    module, asset_root, _ = parent_tree
    public, private = module.build_bindings(asset_root)
    encoded = json.dumps(public)
    assert str(asset_root) not in encoded and "targets" not in encoded
    assert set(public) == {"92601", "92602"}
    assert public["92601"]["GT"]["historical_protocol"]["source"] == "source_snapshots/protocol_memory_retry_v3/protocol_v2.json"
    assert private["92601"]["GT"]["arrays"]["CALIBRATED_F"].shape == (371, 48, 21)
    assert private["92601"]["TS2"]["decoder_checkpoint"].is_file()


def test_bindings_reject_tampered_prediction(parent_tree):
    module, asset_root, _ = parent_tree
    path = asset_root / ".cache/professor_compressor_direct_20261011/runs/jena_GT_full_s92601/prediction.npz"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA"):
        module.build_bindings(asset_root)


def test_missing_ts2_decoder_keeps_encoder_usable(parent_tree):
    module, asset_root, _ = parent_tree
    path = asset_root / ".cache/professor_compressor_direct_20261011/runs/jena_TS2_d320_s92601_retry_batch4_v4/forecast_calibrated_decoder_best.pt"
    path.unlink()
    public, private = module.build_bindings(asset_root)
    assert public["92601"]["TS2"]["decoder_available"] is False
    assert private["92601"]["TS2"]["checkpoint"].is_file()
