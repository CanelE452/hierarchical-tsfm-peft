import copy
import importlib
import json

import numpy as np
import pytest
import torch
from torch import nn

from research.gt_ts2_fastcheck_20261018_v1.model import ForecastRoute, create_gt


def training_module():
    try:
        return importlib.import_module("research.gt_ts2_fastcheck_20261018_v1.training")
    except ModuleNotFoundError as error:
        if error.name == "research.gt_ts2_fastcheck_20261018_v1.training":
            pytest.fail("The fixed-schedule forecast adaptation trainer is missing")
        raise


class FrozenFakeFoundation:
    def __init__(self):
        self.model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.model.weight.fill_(.8)

    def forecast(self, latent, horizon):
        context = latent[:, -1:] + .2 * latent.mean(1, keepdim=True)
        return (context * self.model.weight[0, 0]).expand(-1, horizon, -1, -1)


def route_and_data(count=13, fused=True):
    gt = create_gt(torch.eye(21), seed=92601)
    with torch.no_grad():
        for layer in [*gt.encoder, *gt.decoder]:
            layer.weight.zero_()
            layer.weight[:, 0, 0] = torch.tensor([1., .02, .01, .005, .002])
        gt.encoder[-1].weight[:, 0, 1] = .01
    foundation = FrozenFakeFoundation()
    route = ForecastRoute(gt, foundation, torch.zeros(320) if fused else None,
                          torch.ones(320) if fused else None)
    x = torch.linspace(.1, .7, count * 16 * 21).reshape(count, 16, 21)
    target = torch.linspace(-.3, .9, count * 48 * 21).reshape(count, 48, 21)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[::2, :35, 0] = False
    mask[1::2, :20, 1:5] = False
    features = torch.linspace(.1, 1., count * 4 * 320).reshape(count, 4, 320)
    return route, x, target, mask, features


def test_fixed_schedule_uses_every_index_and_last_small_batch(tmp_path):
    module = training_module()
    route, x, target, mask, features = route_and_data()
    schedule = [np.array([2, 1, 0, 3, 4, 5, 6, 7]), np.array([8, 9, 10, 11, 12]), np.array([1, 7])]
    result = module.train_arm(route, x, target, mask, schedule, feature_cache=features,
                              output_dir=tmp_path / "run", run_id="fixture", seed=92601)
    assert result["updates"] == 3
    assert result["seen_origin_counts"] == [1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1]
    assert result["seen_origin_count"] == 15 and result["unique_seen_origins"] == 13
    assert [row["effective_batch"] for row in result["curves"]] == [8, 5, 2]
    assert [row["update"] for row in result["curves"]] == [1, 2, 3]
    assert all(row["finite"] for row in result["curves"])
    assert result["F_state_unchanged"] and result["F_parameter_gradients_absent"]
    for group in ("encoder", "decoder", "product", "W"):
        assert result["parameter_deltas"][group]["l2"] > 0
        assert result["curves"][0][f"gradient_norm_{group}"] > 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in route.foundation.model.parameters())
    assert not route.foundation.model.training
    assert result["trainable_parameters"] == 244 + 13440


def test_microbatch_uses_full_effective_batch_channel_counts(tmp_path):
    module = training_module()
    route, x, target, mask, features = route_and_data(count=3)
    manual = copy.deepcopy(route)
    optimizer = torch.optim.Adam(manual.parameters(), lr=1e-4, weight_decay=0)
    prediction = manual.forecast(x, features)
    independent = torch.stack([
        (prediction[:, :, channel][mask[:, :, channel]]
         - target[:, :, channel][mask[:, :, channel]]).square().mean()
        for channel in range(21)
    ]).mean()
    expected_loss = independent.item()
    independent.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(list(manual.parameters()), 1.).item()
    expected_gradient = manual.W.weight.grad.clone()
    optimizer.step()
    result = module.train_arm(route, x, target, mask, [np.array([0, 1, 2])],
                              feature_cache=features, output_dir=tmp_path / "micro", run_id="micro",
                              seed=92601, microbatch=1)
    assert result["curves"][0]["train_loss"] == pytest.approx(expected_loss, rel=1e-6)
    assert result["curves"][0]["preclip_norm"] == pytest.approx(expected_norm, rel=1e-6)
    torch.testing.assert_close(route.W.weight.grad, expected_gradient, rtol=2e-5, atol=2e-6)
    for name, value in route.adapter_state().items():
        torch.testing.assert_close(value, manual.adapter_state()[name], rtol=2e-5, atol=2e-6)


def test_checkpoint_zero_and_hundred_capture_state_rng_and_exposure(tmp_path):
    module = training_module()
    route, x, target, mask, features = route_and_data(count=1)
    initial = route.adapter_state()
    schedule = [np.array([0])] * 100
    callbacks = []
    result = module.train_arm(route, x, target, mask, schedule, feature_cache=features,
                              output_dir=tmp_path / "checkpoints", run_id="hundred", seed=92601,
                              callback=lambda update, details: callbacks.append(update))
    assert [row["update"] for row in result["checkpoints"]] == [0, 100]
    checkpoints = [torch.load(row["path"], map_location="cpu", weights_only=False)
                   for row in result["checkpoints"]]
    assert [row["update"] for row in checkpoints] == [0, 100]
    for name, value in initial.items():
        torch.testing.assert_close(checkpoints[0]["adapter_state"][name], value, rtol=0, atol=0)
    assert checkpoints[1]["seen_origin_counts"] == [100]
    assert checkpoints[1]["schedule_sha256"] == result["schedule_sha256"]
    assert checkpoints[0]["torch_rng_state"].dtype == torch.uint8
    assert checkpoints[0]["cuda_rng_state"] is None
    assert checkpoints[1]["optimizer_state"]["state"]
    assert not any("foundation" in key for key in checkpoints[1]["adapter_state"])
    assert callbacks == list(range(0, 101, 10))
    assert len((tmp_path / "checkpoints" / "TRAINING_CURVES.jsonl").read_text().splitlines()) == 100
    assert (tmp_path / "checkpoints" / "TRAINING_SUMMARY.json").is_file()
    assert len(result["schedule_sha256"]) == 64
    with pytest.raises(FileExistsError):
        module.train_arm(route, x, target, mask, schedule, feature_cache=features,
                         output_dir=tmp_path / "checkpoints", run_id="hundred", seed=92601)


def test_fixed_diagnostics_score_forecast_and_reconstruction_independently(tmp_path):
    module = training_module()
    route, x, target, mask, features = route_and_data(count=2)
    input_mask = torch.ones_like(x, dtype=torch.bool)
    input_mask[:, 5:8, 0] = False
    with torch.no_grad():
        prediction = route.forecast(x, features).numpy().astype(np.float64)
        reconstruction = route.gt.decode(route.gt.encode(x)).numpy().astype(np.float64)
    error = prediction - target.numpy().astype(np.float64)
    observed = mask.numpy()
    expected_forecast = ((np.where(observed, error * error, 0).sum((0, 1))
                          / observed.sum((0, 1))).mean())
    history_error = reconstruction[:, 3:] - x[:, 3:].numpy().astype(np.float64)
    history_observed = input_mask[:, 3:].numpy()
    expected_reconstruction = ((np.where(history_observed, history_error ** 2, 0).sum((0, 1))
                               / history_observed.sum((0, 1))).mean())
    result = module.train_arm(route, x, target, mask, [np.array([1, 0])], feature_cache=features,
                              diagnostics=dict(x=x, targets=target, mask=mask, features=features,
                                               input_mask=input_mask), output_dir=tmp_path / "diag",
                              run_id="diag", seed=92601)
    assert len(result["diagnostics"]) == 1
    diagnostic = result["diagnostics"][0]
    assert diagnostic["update"] == 0
    assert diagnostic["forecast_mse"] == pytest.approx(expected_forecast, abs=1e-12)
    assert diagnostic["reconstruction_mse"] == pytest.approx(expected_reconstruction, abs=1e-12)
    assert diagnostic["reconstruction_excluded_prefix"] == 3
    assert diagnostic["origin_count"] == 2


def test_unobserved_nan_targets_do_not_contaminate_gradients(tmp_path):
    module = training_module()
    route, x, target, mask, features = route_and_data(count=2)
    target[~mask] = float("nan")
    result = module.train_arm(route, x, target, mask, [np.array([0, 1])], feature_cache=features,
                              output_dir=tmp_path / "missing", run_id="missing", seed=92601)
    assert result["curves"][0]["finite"]
    assert torch.isfinite(route.W.weight).all()


def test_numpy_inputs_and_gt_only_arm_keep_full_channel_macro_weighting(tmp_path):
    module = training_module()
    route, x, target, mask, _ = route_and_data(count=2, fused=False)
    mask[:, :, 20] = False
    target[:, :, 20] = float("nan")
    with torch.no_grad():
        prediction = route.forecast(x)
        expected = torch.stack([
            (prediction[:, :, channel][mask[:, :, channel]]
             - target[:, :, channel][mask[:, :, channel]]).square().mean()
            for channel in range(20)
        ]).mean().item()
    result = module.train_arm(route, x.numpy(), target.numpy(), mask.numpy(), [np.array([1, 0])],
                              output_dir=tmp_path / "numpy", run_id="numpy", seed=92601, microbatch=2)
    assert result["trainable_parameters"] == 244
    assert result["curves"][0]["active_channels"] == 20
    assert result["curves"][0]["train_loss"] == pytest.approx(expected, rel=1e-6)
    assert result["parameter_deltas"]["W"]["changed_elements"] == 0


@pytest.mark.parametrize("bad", ["observed_nan", "numeric_mask", "no_observations", "bad_schedule"])
def test_invalid_training_contract_fails_before_checkpoint(tmp_path, bad):
    module = training_module()
    route, x, target, mask, features = route_and_data(count=2)
    schedule = [np.array([0, 1])]
    if bad == "observed_nan":
        target[0, -1, 0] = float("nan")
    elif bad == "numeric_mask":
        mask = mask.float()
    elif bad == "no_observations":
        mask.zero_()
    else:
        schedule = [np.array([0] * 9)]
    with pytest.raises(ValueError):
        module.train_arm(route, x, target, mask, schedule, feature_cache=features,
                         output_dir=tmp_path / "invalid", run_id="invalid", seed=92601)
    assert not list(tmp_path.rglob("*.pt"))


def test_callback_failure_preserves_attempt_checkpoints_and_curves(tmp_path):
    module = training_module()
    route, x, target, mask, features = route_and_data(count=1)

    def stop(update, details):
        if update == 10:
            raise RuntimeError("fixture budget stop")

    with pytest.raises(RuntimeError, match="fixture budget stop"):
        module.train_arm(route, x, target, mask, [np.array([0])] * 12, feature_cache=features,
                         output_dir=tmp_path / "stopped", run_id="stopped", seed=92601, callback=stop)
    assert len(list((tmp_path / "stopped").glob("*.pt"))) == 1
    rows = [json.loads(line) for line in (tmp_path / "stopped" / "TRAINING_CURVES.jsonl").read_text().splitlines()]
    assert [row["update"] for row in rows] == list(range(1, 11))
    assert not (tmp_path / "stopped" / "TRAINING_SUMMARY.json").exists()
