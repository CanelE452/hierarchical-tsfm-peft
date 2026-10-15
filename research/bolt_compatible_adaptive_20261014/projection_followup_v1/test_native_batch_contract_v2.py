"""Regression fixtures for the exact native checkpoint prediction artifact."""
from types import SimpleNamespace

import numpy as np
import torch

import native_head_v1 as nh
from native_batch_contract_v2 import native_components


def fixture():
    rng = np.random.default_rng(92601)
    state = SimpleNamespace(center=rng.normal(size=17), encoder=rng.normal(size=(17, 4)),
                            decoder=rng.normal(size=(4, 17)), decoder_bias=np.zeros(17))
    history = rng.normal(size=(65, 512, 17)).astype(np.float32)
    model = nh.NativeForecaster(state)
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy(rng.normal(0, .001, size=(4, 48, 512)).astype(np.float32)))
        model.head.bias.copy_(torch.from_numpy(rng.normal(0, .001, size=(4, 48)).astype(np.float32)))
    model.eval().requires_grad_(False)
    return model, history


def test_saved_prediction_is_exact_checkpoint_scoring_path_including_tail():
    model, history = fixture()
    reference_latent = nh.cache_latents(model, {"history": history}, "fixture", None)
    reference = nh.predict_cached_latents(model, reference_latent, history[:, -1:])
    prediction, latent, future = native_components(model, history)
    np.testing.assert_array_equal(prediction, reference)
    np.testing.assert_array_equal(latent, reference_latent.numpy())
    with torch.no_grad():
        expected_future = torch.cat([model.head(reference_latent[start:start + 32])
                                     for start in range(0, len(history), 32)]).numpy()
    np.testing.assert_array_equal(future, expected_future)
    assert all(value.flags.c_contiguous for value in (prediction, latent, future))


def test_zero_head_keeps_original_current_C_anchor_exact():
    model, history = fixture()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
    prediction, latent, future = native_components(model, history)
    np.testing.assert_array_equal(prediction, np.broadcast_to(history[:, -1:], prediction.shape))
    np.testing.assert_array_equal(future, np.broadcast_to(latent[:, -1:], future.shape))
