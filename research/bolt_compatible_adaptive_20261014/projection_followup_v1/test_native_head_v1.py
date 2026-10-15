"""Synthetic fixtures only; no real data, checkpoint, or forecaster asset reads."""

import inspect
import io
from types import SimpleNamespace

import numpy as np
import torch

from native_head_v1 import NativeForecaster, NativeHead, masked_macro_mse
from native_head_v1 import predict_cached_latents, select_checkpoint, train_native


def projection():
    return SimpleNamespace(center=np.array([0.5, -0.25]), encoder=np.eye(2),
                           decoder=np.eye(2), decoder_bias=np.array([0.125, 0.25]))


def payload(seed=31):
    rng = np.random.default_rng(seed)
    history = rng.normal(size=(8, 6, 2)).astype(np.float32)
    target = history[:, -1:] + np.array([[[0.2, -0.1], [0.3, -0.15]]], np.float32)
    return {"history": history, "target": target, "observed": np.ones_like(target, dtype=bool),
            "raw_last": history[:, -1:], "origins": np.arange(8, dtype=np.int64)}


def test_each_latent_uses_only_its_own_temporal_head():
    head = NativeHead(2, L=3, H=2)
    with torch.no_grad():
        head.weight[0, :, 0] = 1
        head.weight[1, :, 1] = 2
    first = torch.tensor([[[1., 10.], [2., 20.], [3., 30.]]])
    changed = first.clone()
    changed[:, :, 1] += torch.tensor([1., 2., 3.])
    torch.testing.assert_close(head(first)[:, :, 0], head(changed)[:, :, 0], rtol=0, atol=0)
    assert not torch.equal(head(first)[:, :, 1], head(changed)[:, :, 1])


def test_masked_nan_targets_do_not_affect_loss_or_gradients():
    prediction = torch.tensor([[[1., 2.], [3., 4.]]], requires_grad=True)
    target = torch.tensor([[[0., float("nan")], [1., float("nan")]]])
    observed = torch.tensor([[[True, False], [True, False]]])
    loss = masked_macro_mse(prediction, target, observed)
    loss.backward()
    assert float(loss.detach()) == 2.5
    torch.testing.assert_close(prediction.grad, torch.tensor([[[1., 0.], [2., 0.]]]))


def test_zero_head_is_exact_current_anchor_with_latent_innovation_first():
    model = NativeForecaster(projection(), L=6, H=2)
    data = payload()
    predicted = model(torch.as_tensor(data["history"])).detach().numpy()
    expected = np.broadcast_to(data["raw_last"], predicted.shape)
    np.testing.assert_array_equal(predicted, expected)
    assert predicted.tobytes() == expected.tobytes()
    assert not any(buffer.requires_grad for buffer in model.projection.buffers())
    assert not list(model.projection.parameters())


def test_checkpoint_exact_tie_selects_earliest_update():
    selected = select_checkpoint([{"update": 200, "tune": {"mse": 0.25}},
                                  {"update": 0, "tune": {"mse": 0.25}},
                                  {"update": 100, "tune": {"mse": 0.5}}])
    assert selected["update"] == 0


def test_actual_training_tie_retains_checkpoint_zero():
    data = payload()
    data["target"] = np.broadcast_to(data["raw_last"], data["target"].shape).copy()
    result = train_native(projection(), data, data, 92601, updates=4)
    assert result["metadata"]["selected_update"] == 0
    assert result["metadata"]["CP0_anchor_max_abs_difference_FP32"] == 0.0
    assert all(checkpoint["tune"]["mse"] == 0.0 for checkpoint in result["checkpoints"])


def test_forward_has_no_future_target_or_teacher_argument():
    assert list(inspect.signature(NativeForecaster.forward).parameters) == ["self", "history", "raw_last"]
    assert list(inspect.signature(NativeHead.forward).parameters) == ["self", "latent_history"]


def test_short_training_freezes_codec_and_serializes_selected_state():
    data = payload()
    events, callbacks = [], []
    result = train_native(projection(), data, data, 92601, events.append,
                          lambda update, state, score: callbacks.append((update, score)), updates=12)
    repeated = train_native(projection(), data, data, 92601, updates=12)
    assert result["metadata"]["encoder_decoder_frozen"]
    assert result["metadata"]["order_sha256"] == repeated["metadata"]["order_sha256"]
    assert result["metadata"]["selected_state_sha256"] == repeated["metadata"]["selected_state_sha256"]
    assert len(result["learning_curve"]) == 12
    assert [item[0] for item in callbacks] == [0, 12]
    for key in ("center", "encoder", "decoder", "decoder_bias"):
        np.testing.assert_array_equal(getattr(result["model"].projection, key).numpy(),
                                      np.asarray(getattr(projection(), key), dtype=np.float32))
    buffer = io.BytesIO()
    torch.save(result["selected_state"], buffer)
    buffer.seek(0)
    reloaded = NativeForecaster(projection(), L=6, H=2)
    reloaded.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
    with torch.no_grad():
        expected = result["model"](torch.as_tensor(data["history"])).numpy()
        actual = reloaded(torch.as_tensor(data["history"])).numpy()
        latent = reloaded.projection.encode(torch.as_tensor(data["history"]))
    np.testing.assert_array_equal(expected, actual)
    np.testing.assert_array_equal(actual, predict_cached_latents(reloaded, latent, data["raw_last"], batch_size=3))
