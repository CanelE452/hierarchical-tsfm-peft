"""Synthetic fixed-D head/CP/mask fixtures; no scientific assets or large training."""
import inspect

import numpy as np
import pytest
import torch

import raw_subset_native_head_v1 as native


def payload(n=8, seed=907):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 512, 2)).astype(np.float32)
    last = rng.normal(size=(n, 3)).astype(np.float32)
    target = np.repeat(last[:, None], 48, axis=1)+np.float32(.1)
    mask = np.ones(target.shape, bool); mask[:, :7, 1] = False; target[~mask] = np.nan
    return {"latent_history": z, "raw_last": last, "target": target, "observed": mask, "origins": np.arange(n, dtype=np.int64)}


def decoder():
    return np.array([[1., 0., .3], [0., 1., .4]], np.float32)


def test_zero_head_numeric_anchor_fixed_matrix_and_perlatent_head():
    model = native.NativeFixedDecoder(decoder()); p = payload()
    actual, _ = native.predict_components(model, p["latent_history"], p["raw_last"])
    np.testing.assert_array_equal(actual, np.repeat(p["raw_last"][:, None], 48, axis=1))
    assert set(dict(model.named_parameters())) == {"head.weight", "head.bias"}
    assert not model.decoder.requires_grad and model.decoder.shape == (2, 3)
    with torch.no_grad(): model.head.bias[0] = 1.
    actual, _ = native.predict_components(model, p["latent_history"], p["raw_last"])
    np.testing.assert_allclose(actual-p["raw_last"][:, None], np.broadcast_to(decoder()[0], actual.shape), atol=3e-7, rtol=0)


def test_inputs_and_decoder_never_receive_gradients():
    model = native.NativeFixedDecoder(decoder())
    p = payload(4)
    z = torch.from_numpy(p["latent_history"]).requires_grad_(); last = torch.from_numpy(p["raw_last"]).requires_grad_()
    loss = native.masked_macro_mse(model(z, last), torch.from_numpy(p["target"]), torch.from_numpy(p["observed"]))
    loss.backward()
    assert z.grad is last.grad is model.decoder.grad is None
    assert torch.isfinite(model.head.bias.grad).all() and torch.count_nonzero(model.head.bias.grad) > 0


def test_small_training_frozen_D_and_exact_CP_callback_batch32():
    fit, tune = payload(), payload(65, 908); original = {key: value.tobytes() for key, value in fit.items()}
    captured = []
    def callback(update, state, score, prediction):
        model = native.NativeFixedDecoder(decoder()); model.load_state_dict(state)
        actual, _ = native.predict_components(model, tune["latent_history"], tune["raw_last"])
        assert prediction.tobytes() == actual.tobytes()
        assert score == native.score_predictions(prediction, tune["target"], tune["observed"])
        captured.append(update)
    result = native.train_native_fixed(decoder(), fit, tune, 92601, checkpoint_callback=callback, updates=8)
    assert captured == [0, 8] and len(result["learning_curve"]) == 8
    meta = result["metadata"]
    assert meta["fixed_decoder_before_sha256"] == meta["fixed_decoder_after_sha256"]
    assert meta["fixed_global_FIT_decoder"] and not meta["per_origin_decoder_fit_or_rho_selection"]
    assert meta["CP0_currentC_numeric_max_abs_difference"] == 0
    assert all(fit[key].tobytes() == value for key, value in original.items())
    assert all(np.isfinite(row["loss"]) and np.isfinite(row["gradient_norm_before_clip"]) for row in result["learning_curve"])


def test_serialization_and_two_seed_zero_initialization_contract():
    fit, tune = payload(), payload(7, 909)
    a = native.train_native_fixed(decoder(), fit, tune, 92601, updates=3)
    b = native.train_native_fixed(decoder(), fit, tune, 92602, updates=3)
    assert a["metadata"]["initial_head_sha256"] == b["metadata"]["initial_head_sha256"]
    assert a["metadata"]["order_sha256"] != b["metadata"]["order_sha256"]
    copy = native.NativeFixedDecoder(decoder()); copy.load_state_dict(a["selected_state"])
    x, _ = native.predict_components(a["model"], tune["latent_history"], tune["raw_last"])
    y, _ = native.predict_components(copy, tune["latent_history"], tune["raw_last"])
    assert x.tobytes() == y.tobytes()
    assert set(inspect.signature(copy.forward).parameters) == {"latent_history", "raw_last"}


def test_loss_observed_C_macro_and_exact_tie_rule_match_original():
    p = torch.tensor([[[1., 2.]], [[3., 4.]]], requires_grad=True)
    y = torch.tensor([[[0., float("nan")]], [[0., 0.]]]); m = torch.tensor([[[True, False]], [[True, True]]])
    loss = native.masked_macro_mse(p, y, m); assert loss.item() == 10.5
    loss.backward(); torch.testing.assert_close(p.grad, torch.tensor([[[.5, 0.]], [[1.5, 4.]]]), rtol=0, atol=0)
    assert native.select_checkpoint([{"update": 200, "tune": {"mse": 1}}, {"update": 0, "tune": {"mse": 1}}])["update"] == 0


@pytest.mark.parametrize("change", ["D_FP64", "D_nonfinite", "z_FP64", "Y_observed_NaN", "origins_int32", "no_observations"])
def test_invalid_fixed_training_contract_rejected(change):
    d = decoder(); fit, tune = payload(), payload(7, 910)
    if change == "D_FP64": d = d.astype(np.float64)
    elif change == "D_nonfinite": d[0, 0] = np.inf
    elif change == "z_FP64": fit["latent_history"] = fit["latent_history"].astype(np.float64)
    elif change == "Y_observed_NaN": fit["target"][0, 0, 0] = np.nan
    elif change == "origins_int32": fit["origins"] = fit["origins"].astype(np.int32)
    else: fit["observed"][:] = False
    with pytest.raises(ValueError): native.train_native_fixed(d, fit, tune, 92601, updates=2)
