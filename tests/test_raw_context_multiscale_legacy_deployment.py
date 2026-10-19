"""Legacy Raw identity checks through the real physical CPU deployment path."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.informs_broad_validation_20261015_v1.gpu_runtime import RowForecaster
from research.raw_context_multiscale_20261018_v1 import common, deployment


PINNED_F = "c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874"


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def task():
    values = np.arange(144 * 4, dtype=np.float32).reshape(144, 4) / 100
    return dict(physical=values, observed=np.ones_like(values, bool), L=64, H=8,
                columns=["a", "b", "c", "d"], data_sha256="synthetic-train-only",
                split_ranges=dict(A_fit=(0, 80), A_stop=(80, 96), M_cal=(96, 112),
                                  V_select=(112, 128), T_final=(128, 144)))


def context(tmp_path):
    return dict(asset=tmp_path, cache=tmp_path, device="cpu",
                modules=dict(legacy=SimpleNamespace(F_SHA=PINNED_F, MODEL_REL="unused-snapshot")))


def pool(data, identity=PINNED_F):
    value = deployment.legacy.FoundationPool(None, "cpu")
    value.foundation = RowForecaster(
        lambda rows: rows[:, -1:] + torch.arange(data["H"], dtype=rows.dtype)[None] / 10,
        data["H"], 128, False)
    value.foundation.initial_hash = identity
    value.instances_created = 1
    assert value.factory is None
    return value


def test_unpinned_legacy_raw_is_bound_on_copy_and_runs_whole_physical_path(tmp_path):
    data, ctx = task(), context(tmp_path)
    spec = dict(family="RAW", run_id="Raw_fixture", canonical_fit_end=96)
    original = copy.deepcopy(spec)
    shared = pool(data)
    route = deployment.make_route(ctx, data, spec, foundation_pool=shared)
    assert route.spec["foundation_sha"] == PINNED_F
    assert spec == original
    assert route.pool is shared and route.pool.factory is None
    physical = data["physical"][:64][None].copy()
    observed = data["observed"][:64][None].copy()
    observed[0, 20:23, 2] = False
    physical[0, 20:23, 2] = np.nan
    prediction = route.predict_physical(physical, observed, measure_stages=True)
    expected = physical[:, -1:].astype(np.float64) + np.arange(8)[None, :, None] / 10 * route.std
    np.testing.assert_allclose(prediction, expected, atol=5e-7, rtol=1e-6)
    assert route.last_receipt["F_calls"] == 1
    assert route.last_receipt["F_invocations"][0]["input_shape"] == [1, 64, 4]
    assert {"CPU_preprocess", "host_to_device", "F", "inverse_and_CPU"} <= set(route.last_receipt["stage_seconds"])
    route.close()
    assert shared.foundation is not None


def test_explicit_raw_pin_is_preserved_instead_of_replaced_by_context(tmp_path):
    data, ctx = task(), context(tmp_path)
    explicit = "a" * 64
    spec = dict(family="RAW", foundation_sha=explicit)
    route = deployment.make_route(ctx, data, spec, foundation_pool=pool(data, explicit))
    assert route.spec["foundation_sha"] == explicit
    assert spec == dict(family="RAW", foundation_sha=explicit)
    route.close()


@pytest.mark.parametrize("explicit", ["", None])
def test_explicit_empty_raw_pin_is_not_repaired_into_a_valid_identity(tmp_path, explicit):
    data, ctx = task(), context(tmp_path)
    spec = dict(family="RAW", foundation_sha=explicit)
    with pytest.raises(ValueError, match="pinned foundation identity"):
        deployment.make_route(ctx, data, spec, foundation_pool=pool(data))
    assert spec == dict(family="RAW", foundation_sha=explicit)


@pytest.mark.parametrize("explicit", [None, "b" * 64])
def test_raw_identity_mismatch_still_fails_in_inherited_check(tmp_path, explicit):
    data, ctx = task(), context(tmp_path)
    spec = dict(family="RAW")
    if explicit is not None:
        spec["foundation_sha"] = explicit
    original = copy.deepcopy(spec)
    actual = "a" * 64 if explicit is None else PINNED_F
    with pytest.raises(ValueError, match="foundation identity differs"):
        deployment.make_route(ctx, data, spec, foundation_pool=pool(data, actual))
    assert spec == original


def test_explicit_official_raw_asset_remains_the_identity_authority(tmp_path):
    data, ctx = task(), context(tmp_path)
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    asset = dict(files={config.name: dict(sha256=common.sha(config))})
    spec = dict(family="RAW", foundation_asset=asset)
    shared = pool(data, "a" * 64)
    shared.model_path = tmp_path
    route = deployment.make_route(ctx, data, spec, foundation_pool=shared)
    assert "foundation_sha" not in route.spec
    assert route.spec["foundation_asset"] == asset
    assert spec == dict(family="RAW", foundation_asset=asset)
    route.close()


def test_combined_nested_raw_gets_same_pin_without_mutating_parent_spec(tmp_path):
    data, ctx = task(), context(tmp_path)
    spec = dict(family="COMBINED", operation="alpha", alpha=.5,
                g_route=dict(family="CURRENT"), b_route=dict(family="RAW"))
    original = copy.deepcopy(spec)
    shared = pool(data)
    route = deployment.make_route(ctx, data, spec, foundation_pool=shared)
    assert route.b_route.spec["foundation_sha"] == PINNED_F
    assert "foundation_sha" not in route.g_route.spec
    assert route.b_route.pool is shared and route.g_route.pool is shared
    assert spec == original
    physical = data["physical"][:64][None].copy()
    prediction = route.predict_physical(physical, data["observed"][:64][None])
    expected = physical[:, -1:].astype(np.float64) + np.arange(8)[None, :, None] / 20 * route.std
    np.testing.assert_allclose(prediction, expected, atol=5e-7, rtol=1e-6)
    assert route.last_receipt["F_calls"] == 1
    route.close()


def test_direct_g_legacy_bundle_does_not_bind_or_load_foundation(tmp_path):
    data, ctx = task(), context(tmp_path)
    fit = deployment.legacy.fit_base
    base_spec = dict(family="DIRECT_G", seed=92601, device="cpu")
    model, normalization = fit.create_model(base_spec, data, 96, None)
    state = model.state_dict()
    state_sha = deployment.legacy.fingerprint({
        key: deployment.legacy.array_sha256(value.numpy()) for key, value in state.items()})
    path = tmp_path / "synthetic_direct_g.pt"
    torch.save(dict(spec=base_spec, model=state, model_state_sha=state_sha,
                    normalization=normalization,
                    contract=dict(data_sha=data["data_sha256"], model_sha="NO_FOUNDATION",
                                  scaler_sha=fit.scaler_fingerprint(normalization["mean"], normalization["std"]))), path)
    spec = dict(family="DIRECT_G", artifact_cache_relative_path=path.name,
                artifact_sha256=common.sha(path), canonical_fit_end=96)
    original = copy.deepcopy(spec)
    shared = deployment.legacy.FoundationPool(None, "cpu")
    route = deployment.make_route(ctx, data, spec, foundation_pool=shared)
    prediction = route.predict_physical(data["physical"][:64][None], data["observed"][:64][None])
    assert prediction.shape == (1, 8, 4) and np.isfinite(prediction).all()
    assert spec == original and "foundation_sha" not in route.spec
    assert shared.foundation is None and shared.instances_created == 0
    assert route.last_receipt["F_calls"] == 0
    route.close()
