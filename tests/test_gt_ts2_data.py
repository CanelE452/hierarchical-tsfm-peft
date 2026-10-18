import copy
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch


def data_module():
    try:
        return importlib.import_module("research.gt_ts2_fastcheck_20261018_v1.data")
    except ModuleNotFoundError:
        pytest.fail("The TRAIN/VAL-only fastcheck data implementation is missing")


def fixture_data():
    physical = np.arange(200, dtype=np.float32).reshape(100, 2)
    mask = np.ones_like(physical, dtype=bool)
    physical[5:8, 0] = np.nan
    mask[5:8, 0] = False
    mean = np.array([10, 20], dtype=np.float32)
    std = np.array([2, 4], dtype=np.float32)
    filled = physical.copy()
    filled[5:8, 0] = 8
    return {
        "raw_physical": physical,
        "mask": mask,
        "mean": mean,
        "std": std,
        "legacy_x": ((filled - mean) / std).astype(np.float32),
        "train_end": 48,
        "val_end": 100,
    }


def test_fill_initializes_each_window_from_train_mean_and_keeps_observed_zero():
    module = data_module()
    physical = np.array([[[np.nan, 0], [14, np.nan], [np.nan, 28]],
                         [[np.nan, np.nan], [6, 12], [np.nan, np.nan]]], np.float32)
    mask = np.isfinite(physical)
    actual = module.fill_scale(physical, mask, np.array([10, 20]), np.array([2, 4]))
    expected = np.array([[[0, -5], [2, -5], [2, 2]],
                         [[0, 0], [-2, -2], [-2, -2]]], np.float32)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32


def test_fill_all_missing_channel_is_zero_and_unobserved_infinity_is_ignored():
    module = data_module()
    actual = module.fill_scale(np.array([[np.nan, np.inf], [np.nan, -np.inf]]),
                               np.zeros((2, 2), bool), np.array([3, 4]), np.array([2, 5]))
    np.testing.assert_array_equal(actual, np.zeros((2, 2), np.float32))


@pytest.mark.parametrize("change", ["rank", "mask_shape", "mask_dtype", "mean_shape", "std_shape",
                                    "zero_std", "nonfinite_mean", "observed_nonfinite", "empty"])
def test_fill_rejects_invalid_shapes_or_observed_values(change):
    module = data_module()
    x, mask = np.ones((3, 2)), np.ones((3, 2), bool)
    mean, std = np.zeros(2), np.ones(2)
    if change == "rank":
        x, mask = x.ravel(), mask.ravel()
    elif change == "mask_shape":
        mask = mask[:, :1]
    elif change == "mask_dtype":
        mask = mask.astype(np.int64)
    elif change == "mean_shape":
        mean = mean[None]
    elif change == "std_shape":
        std = std[:1]
    elif change == "zero_std":
        std[0] = 0
    elif change == "nonfinite_mean":
        mean[0] = np.nan
    elif change == "observed_nonfinite":
        x[0, 0] = np.inf
    elif change == "empty":
        x, mask = x[:0], mask[:0]
    with pytest.raises(ValueError):
        module.fill_scale(x, mask, mean, std)


def test_window_start_cannot_receive_a_value_before_its_own_history():
    module = data_module()
    data = fixture_data()
    short, target, target_mask = module.windows(data, [8], 3)
    long, _, _ = module.windows(data, [8], 4)
    old, old_target, old_mask = module.windows(data, [8], 3, legacy=True)
    np.testing.assert_array_equal(short[0, :, 0], [0, 0, 0])
    np.testing.assert_array_equal(long[0, :, 0], [-1, -1, -1, -1])
    np.testing.assert_array_equal(old[0, :, 0], [-1, -1, -1])
    np.testing.assert_array_equal(target, old_target)
    np.testing.assert_array_equal(target_mask, old_mask)
    data["raw_physical"][4, 0] = 99999
    np.testing.assert_array_equal(module.windows(data, [8], 3)[0], short)
    assert short.dtype == target.dtype == np.float32
    assert target_mask.dtype == bool


def test_future_changes_cannot_change_inference_history():
    module = data_module()
    data = fixture_data()
    original = module.windows(data, [10], 5)[0]
    changed = copy.deepcopy(data)
    changed["raw_physical"][10:] = 1e10
    changed["legacy_x"][10:] = -1e10
    changed["mask"][10:] = False
    np.testing.assert_array_equal(module.windows(changed, [10], 5)[0], original)


def test_window_batch_and_channel_shuffle_preserve_values():
    module = data_module()
    data = fixture_data()
    x, y, mask = module.windows(data, [8, 10, 12], 3)
    batch_order = [2, 0, 1]
    shuffled = copy.deepcopy(data)
    for key in ("raw_physical", "mask", "legacy_x"):
        shuffled[key] = shuffled[key][:, [1, 0]]
    for key in ("mean", "std"):
        shuffled[key] = shuffled[key][[1, 0]]
    actual = module.windows(shuffled, [12, 8, 10], 3)
    for got, expected in zip(actual, (x, y, mask)):
        np.testing.assert_array_equal(got, expected[batch_order][:, :, [1, 0]])


def test_physical_window_is_exact_history_and_empty_batch_has_explicit_shape():
    module = data_module()
    data = fixture_data()
    physical, mask = module.physical_windows(data, [8], 3)
    np.testing.assert_array_equal(physical, data["raw_physical"][5:8][None])
    np.testing.assert_array_equal(mask, data["mask"][5:8][None])
    physical[0, 0, 1] = 1234
    assert data["raw_physical"][5, 1] == 11
    x, y, target_mask = module.windows(data, np.array([], np.int64), 3)
    assert x.shape == (0, 3, 2)
    assert y.shape == target_mask.shape == (0, 48, 2)


@pytest.mark.parametrize("origins,length", [([2], 3), ([53], 3), ([8.5], 3), ([[8]], 3),
                                            ([True], 3), ([8], 0), ([8], 3.5)])
def test_windows_reject_out_of_bounds_or_noninteger_requests(origins, length):
    with pytest.raises(ValueError):
        data_module().windows(fixture_data(), origins, length)


def test_channel_macro_loss_uses_equal_channel_weight_not_equal_observation_weight():
    module = data_module()
    pred = torch.tensor([[[2., 3.], [4., 99.]]], requires_grad=True)
    target = torch.tensor([[[0., 0.], [0., float("nan")]]])
    mask = torch.tensor([[[True, True], [True, False]]])
    loss = module.channel_macro_loss(pred, target, mask)
    assert loss.item() == 9.5
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.tensor([[[1., 3.], [2., 0.]]]))


def test_microbatch_losses_and_gradients_sum_to_the_full_channel_macro_loss():
    module = data_module()
    pred = torch.tensor([[[1., 3.], [2., 4.]], [[5., 6.], [7., 8.]],
                         [[9., 10.], [11., 12.]]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[[True, False], [True, False]], [[True, True], [False, True]],
                         [[False, True], [True, False]]])
    target = torch.zeros_like(pred)
    full = module.channel_macro_loss(pred, target, mask)
    full.backward()
    full_grad = pred.grad.clone()
    split_pred = pred.detach().clone().requires_grad_()
    counts = mask.sum((0, 1))
    split = sum(module.channel_macro_loss(split_pred[i:i + 1], target[i:i + 1], mask[i:i + 1],
                                          global_counts=counts) for i in range(3))
    split.backward()
    torch.testing.assert_close(split, full, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(split_pred.grad, full_grad, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("change", ["shape", "mask_dtype", "all_missing_channel", "bad_prediction",
                                    "bad_observed_target", "global_shape", "global_zero", "global_small"])
def test_channel_macro_loss_rejects_invalid_effective_batch(change):
    module = data_module()
    pred, target = torch.ones(1, 2, 2), torch.zeros(1, 2, 2)
    mask = torch.ones(1, 2, 2, dtype=torch.bool)
    counts = None
    if change == "shape":
        target = target[:, :1]
    elif change == "mask_dtype":
        mask = mask.float()
    elif change == "all_missing_channel":
        mask[:, :, 0] = False
    elif change == "bad_prediction":
        pred[0, 0, 0] = float("nan")
    elif change == "bad_observed_target":
        target[0, 0, 0] = float("inf")
    elif change == "global_shape":
        counts = torch.tensor([4])
    elif change == "global_zero":
        counts = torch.tensor([0, 2])
    elif change == "global_small":
        counts = torch.tensor([1, 2])
    with pytest.raises(ValueError):
        module.channel_macro_loss(pred, target, mask, global_counts=counts)


def test_missing_archive_fails_without_creating_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        data_module().load_data(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.fixture(scope="module")
def actual_data():
    asset_root = os.environ.get("GT_TS2_ASSET_ROOT")
    if not asset_root:
        pytest.skip("Set GT_TS2_ASSET_ROOT explicitly for the CPU TRAIN/VAL fingerprint check")
    return data_module().load_data(Path(asset_root))


def test_actual_trainval_arrays_preserve_all_legacy_fingerprints(actual_data):
    module = data_module()
    expected = {
        "legacy_x": "1681c7d0aa3deb7ea784921e34176440fe6bff8c71f2810e0a4707e551a539d3",
        "mask": "43354f82269ee9d72680b6fe5ec06ab7e331762eef9cdbc631ae99007932423f",
        "raw_physical": "c2664a9b82360b7791094b828179129341eab4468b89c456bbb779664c6f6add",
        "mean": "0b2d6ed2ea90ef8d326f06a572cc07fea91f9a46a4f386d72002b6c16820692f",
        "std": "e1875d31b7ac944bcc6e0c72a292b898ede117834b19cd8ae5930c7a7f450037",
        "val_origins": "6493583a54ec2a350be4834e88642b893b182985cfa6b85b7eb52393d6f8f09b",
        "graph": "ffcb0e1bf71229db00a97303f167642d63dd15177fa60ea6bedcf36cadd5fd23",
    }
    for key, digest in expected.items():
        assert module.array_hash(actual_data[key]) == digest, key
    assert actual_data["legacy_contract_hash"] == "742f7189e27439d282bb4ccc0f7a72c3088146e393f8e12c7e1b72af16fd5d8e"
    assert actual_data["raw_physical"].shape == actual_data["mask"].shape == (35135, 21)
    assert actual_data["train_end"] == 26207
    assert actual_data["val_end"] == 35135
    assert np.sum(actual_data["mask"] & (actual_data["raw_physical"] == 0)) == 113247


def test_actual_origin_pools_allow_train_history_without_target_spill(actual_data):
    np.testing.assert_array_equal(actual_data["val_origins"], np.arange(26207, 35088, 24))
    np.testing.assert_array_equal(actual_data["train_origins"], np.arange(1024, 26113, 128))
    np.testing.assert_array_equal(actual_data["diagnostic_origins"],
                                  actual_data["train_origins"][[0, 13, 26, 39, 52, 65, 78, 91,
                                                               104, 117, 130, 143, 156, 169, 182, 196]])
    assert len(actual_data["train_origins"]) == 197
    assert len(actual_data["val_origins"]) == 371
    assert np.all(actual_data["val_origins"] - 2048 >= 0)
    assert np.all(actual_data["train_origins"] + 48 <= actual_data["train_end"])
    assert np.all(actual_data["val_origins"] + 48 <= actual_data["val_end"])


def test_actual_per_window_inputs_differ_from_legacy_only_at_missing_boundaries(actual_data):
    module = data_module()
    expected_hashes = {
        512: "cb85fca8d25bf9a122d52c4e551dd9ba348a605888755da67ece3ec26fdac9ed",
        1024: "f159853bf533bf06d0e6829d58b5529237a6e8fcb05762659b7e2ebb10555dad",
        2048: "0a9d00e85e4ab00d11b33104e7782097367f59e726b3288112f6dbbb269d3260",
    }
    largest = 0.
    for length in (512, 1024, 2048):
        current, y, mask = module.windows(actual_data, actual_data["val_origins"], length)
        assert module.array_hash(current) == expected_hashes[length]
        old, old_y, old_mask = module.windows(actual_data, actual_data["val_origins"], length, legacy=True)
        observed = np.stack([actual_data["mask"][origin - length:origin]
                             for origin in actual_data["val_origins"]])
        np.testing.assert_array_equal(current[observed], old[observed])
        np.testing.assert_array_equal(y, old_y)
        np.testing.assert_array_equal(mask, old_mask)
        assert np.isfinite(current).all()
        largest = max(largest, float(np.max(np.abs(current - old))))
    assert largest == pytest.approx(3.043522, abs=1e-6)


def test_actual_port_matches_the_current_read_only_legacy_loader_bitwise(actual_data, monkeypatch):
    from research.professor_compressor_direct_20261011 import data_io as legacy

    with monkeypatch.context() as context:
        context.setattr(legacy._runtime_v18(), "ROOT", Path(os.environ["GT_TS2_ASSET_ROOT"]))
        old = legacy.load_data("jena")
    for key, old_key in (("raw_physical", "raw_physical"), ("legacy_x", "x"), ("mask", "mask"),
                         ("mean", "mean"), ("std", "std"), ("graph", "graph"),
                         ("val_origins", "val_origins")):
        assert actual_data[key].tobytes() == old[old_key].tobytes(), key


def test_public_contract_is_fingerprinted_trainval_only_and_has_no_local_absolute_paths(actual_data):
    contract = data_module().public_contract(actual_data)
    encoded = json.dumps(contract, sort_keys=True)
    assert contract["contract_hash"] == actual_data["contract_hash"]
    assert contract["normalization_fit_scope"] == "TRAIN observed values only"
    assert contract["graph_fit_scope"] == "TRAIN observed pairwise rows only"
    assert contract["protected_test_accessed"] is False
    assert contract["historical_raw_physical_sha256"] == "0c4932c7a35a01fc955db8d9d6add8b575baaac0ce791dc9ef42eb0b116be209"
    assert contract["array_hashes"]["raw_physical"] == "c2664a9b82360b7791094b828179129341eab4468b89c456bbb779664c6f6add"
    assert contract["origin_counts"] == {"train": 197, "val": 371, "diagnostic": 16}
    assert "\\\\" not in encoded and ":/" not in encoded and ":\\" not in encoded
