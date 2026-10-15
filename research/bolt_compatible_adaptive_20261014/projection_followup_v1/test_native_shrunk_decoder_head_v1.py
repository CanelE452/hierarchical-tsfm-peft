"""Synthetic-only head/gradient/pairing/CP fixtures; no real assets or GPU jobs."""
import inspect

import numpy as np
import pytest
import torch

import native_shrunk_decoder_head_v1 as native


def payload(n=10, seed=7021):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 512, 2)).astype(np.float32)
    decoder = rng.normal(size=(n, 2, 3)).astype(np.float32)
    current = rng.normal(size=(n, 3)).astype(np.float32)
    target = np.repeat(current[:, None], 48, axis=1) + np.float32(.1)
    observed = np.ones(target.shape, bool)
    observed[::2, :9, 1] = False; target[~observed] = np.nan
    return {"latent_history": latent, "decoder": decoder, "raw_last": current, "target": target,
            "observed": observed, "origins": np.arange(500, 500 + n, dtype=np.int64)}


def test_zero_head_is_current_C_and_only_head_parameters_exist():
    values = payload(4)
    model = native.NativeShrunkForecaster(2)
    prediction = native.predict_cached_latents(model, values["latent_history"], values["raw_last"], values["decoder"])
    np.testing.assert_array_equal(prediction, np.repeat(values["raw_last"][:, None], 48, axis=1))
    assert set(dict(model.named_parameters())) == {"head.weight", "head.bias"}
    assert torch.count_nonzero(model.head.weight) == torch.count_nonzero(model.head.bias) == 0


def test_signed_zero_numeric_anchor_is_not_claimed_bit_identical():
    model = native.NativeShrunkForecaster(1, L=2, H=1)
    z = torch.zeros((1, 2, 1)); current = torch.tensor([[-0.]], dtype=torch.float32); d = torch.ones((1, 1, 1))
    prediction = model(z, current, d)
    assert torch.equal(prediction, current[:, None])
    assert prediction.detach().numpy().view(np.uint32)[0, 0, 0] != current.numpy().view(np.uint32)[0, 0]


def test_each_latent_has_an_own_affine_head():
    model = native.NativeShrunkForecaster(2, L=3, H=1)
    with torch.no_grad(): model.head.weight[0, 0, 0] = 2.
    z = torch.tensor([[[1., 9.], [3., 7.], [0., 4.]]])
    prediction = model(z, torch.zeros((1, 2)), torch.eye(2)[None])
    torch.testing.assert_close(prediction, torch.tensor([[[2., 0.]]]), rtol=0, atol=0)


def test_window_decoder_pairing_and_permutation_are_preserved():
    model = native.NativeShrunkForecaster(1, L=2, H=1)
    with torch.no_grad(): model.head.bias.fill_(1.)
    z = torch.zeros((4, 2, 1)); current = torch.tensor([[10.], [20.], [30.], [40.]])
    d = torch.tensor([[[1.]], [[2.]], [[3.]], [[4.]]])
    prediction = model(z, current, d)
    torch.testing.assert_close(prediction[:, 0, 0], torch.tensor([11., 22., 33., 44.]), rtol=0, atol=0)
    order = torch.tensor([3, 1, 0, 2])
    torch.testing.assert_close(model(z[order], current[order], d[order]), prediction[order], rtol=0, atol=0)
    assert not torch.equal(model(z, current, d[order]), prediction)


def test_only_head_receives_gradients_when_cached_inputs_have_requires_grad():
    model = native.NativeShrunkForecaster(1, L=2, H=1)
    z = torch.ones((2, 2, 1), requires_grad=True); current = torch.zeros((2, 1), requires_grad=True)
    d = torch.ones((2, 1, 1), requires_grad=True)
    target = torch.ones((2, 1, 1)); observed = torch.ones(target.shape, dtype=torch.bool)
    native.masked_macro_mse(model(z, current, d), target, observed).backward()
    assert z.grad is current.grad is d.grad is None
    assert torch.isfinite(model.head.bias.grad).all() and torch.count_nonzero(model.head.bias.grad) > 0


def test_macro_loss_uses_observed_C_counts_and_ignores_unobserved_NaNs():
    prediction = torch.tensor([[[1., 2.]], [[3., 4.]]], requires_grad=True)
    target = torch.tensor([[[0., float("nan")]], [[0., 0.]]])
    observed = torch.tensor([[[True, False]], [[True, True]]])
    loss = native.masked_macro_mse(prediction, target, observed)
    assert loss.item() == 10.5
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[[.5, 0.]], [[1.5, 4.]]]), rtol=0, atol=0)
    assert torch.isfinite(prediction.grad).all()


def test_exact_tune_tie_selects_earliest_checkpoint():
    checkpoints = [{"update": 200, "tune": {"mse": .2}}, {"update": 100, "tune": {"mse": .2}}, {"update": 0, "tune": {"mse": .3}}]
    assert native.select_checkpoint(checkpoints)["update"] == 100


def test_finite_toy_training_preserves_all_external_inputs_and_frozen_arrays():
    fit, tune = payload(10), payload(7, 7022)
    before = {name: value.tobytes() for name, value in fit.items()}
    result = native.train_native_shrunk(fit, tune, 92601, updates=8)
    assert len(result["learning_curve"]) == 8 and all(np.isfinite(row["loss"]) and np.isfinite(row["gradient_norm_before_clip"]) for row in result["learning_curve"])
    assert result["metadata"]["frozen_payload_before"] == result["metadata"]["frozen_payload_after"]
    assert result["metadata"]["CP0_anchor_max_abs_difference_FP32"] == 0
    assert all(value.tobytes() == before[name] for name, value in fit.items())
    assert not result["metadata"]["old_head_weights_reused_or_warm_started"]


def test_checkpoint_callback_archives_exact_batch32_scored_array():
    fit, tune = payload(10), payload(65, 7023)
    saved = []
    def callback(update, state, score, prediction):
        replay = native.NativeShrunkForecaster(2); replay.load_state_dict(state); replay.eval()
        actual = native.predict_cached_latents(replay, tune["latent_history"], tune["raw_last"], tune["decoder"], batch_size=32)
        np.testing.assert_array_equal(prediction, actual)
        assert score == native.score_predictions(prediction, tune["target"], tune["observed"])
        saved.append(update)
    result = native.train_native_shrunk(fit, tune, 92601, checkpoint_callback=callback, updates=6)
    assert saved == [0, 6] and result["metadata"]["canonical_checkpoint_prediction_batch_size"] == 32


def test_state_serialization_parity_and_future_targets_absent_from_forward():
    fit, tune = payload(8), payload(6, 7024)
    result = native.train_native_shrunk(fit, tune, 92601, updates=4)
    reconstructed = native.NativeShrunkForecaster(2)
    arrays = {name: value.numpy().copy() for name, value in result["selected_state"].items()}
    reconstructed.load_state_dict({name: torch.from_numpy(value) for name, value in arrays.items()})
    a = native.predict_cached_latents(result["model"], tune["latent_history"], tune["raw_last"], tune["decoder"])
    b = native.predict_cached_latents(reconstructed, tune["latent_history"], tune["raw_last"], tune["decoder"])
    np.testing.assert_array_equal(a, b)
    assert set(inspect.signature(reconstructed.forward).parameters) == {"latent_history", "raw_last", "decoder"}
    assert "dev_payload" not in inspect.signature(native.train_native_shrunk).parameters


def test_seeds_change_batch_order_but_not_zero_initial_weights():
    fit, tune = payload(8), payload(6, 7025)
    a = native.train_native_shrunk(fit, tune, 92601, updates=3)
    b = native.train_native_shrunk(fit, tune, 92602, updates=3)
    assert a["metadata"]["initial_model_sha256"] == b["metadata"]["initial_model_sha256"]
    assert a["metadata"]["order_sha256"] != b["metadata"]["order_sha256"]


@pytest.mark.parametrize("change", ["decoder_wrong_windows", "target_nonfinite_observed", "origins_wrong_dtype", "decoder_FP64"])
def test_invalid_cached_payloads_rejected(change):
    fit, tune = payload(8), payload(6, 7026)
    if change == "decoder_wrong_windows": fit["decoder"] = fit["decoder"][:-1]
    elif change == "target_nonfinite_observed": fit["target"][0, 0, 0] = np.nan
    elif change == "origins_wrong_dtype": fit["origins"] = fit["origins"].astype(np.int32)
    else: fit["decoder"] = fit["decoder"].astype(np.float64)
    with pytest.raises(ValueError): native.train_native_shrunk(fit, tune, 92601, updates=2)


def test_prediction_helper_rejects_silent_FP64_decoder_cast():
    values = payload(4)
    with pytest.raises(ValueError, match="EXACT_FP32"):
        native.predict_cached_latents(native.NativeShrunkForecaster(2), values["latent_history"], values["raw_last"], values["decoder"].astype(np.float64))
