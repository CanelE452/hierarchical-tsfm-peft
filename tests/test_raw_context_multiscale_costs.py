"""Synthetic CPU whole-path checks; never open real data or a GPU."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.raw_context_multiscale_20261018_v1 import common, costs, deployment, models
from research.informs_broad_validation_20261015_v1.gpu_runtime import RowForecaster
from research.informs_broad_validation_20261015_v1.networks import ConditionalHead, ForecastAdapter


@pytest.fixture(autouse=True, scope="module")
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def task():
    rng = np.random.default_rng(123)
    physical = rng.normal(size=(700, 4)).astype(np.float32)
    observed = np.ones_like(physical, dtype=bool)
    observed[22:26, 1] = False
    physical[22:26, 1] = np.nan
    return dict(L=512, H=12, columns=["a", "b", "c", "d"], physical=physical,
                observed=observed, split_ranges={"A_stop": (150, 300), "V_select": (512, 632)})


def pool(task):
    def factory(path, horizon, row_microbatch, device):
        assert horizon == task["H"]
        def predict(rows):
            return rows[:, -1:] + torch.arange(horizon, dtype=rows.dtype)[None]/100
        foundation = RowForecaster(predict, horizon, row_microbatch, False)
        foundation.model = torch.nn.Linear(1, 1).requires_grad_(False)
        foundation.model.chronos_config = SimpleNamespace(context_length=512, prediction_length=64)
        foundation.initial_hash = "fixed-synthetic-F"
        return foundation
    return deployment.legacy.FoundationPool(None, "cpu", factory=factory)


def route(tmp_path, monkeypatch, mode="B", schedule="native", task_value=None):
    data = task_value or task()
    monkeypatch.setattr(deployment.legacy, "fit_statistics",
        lambda t, end: (np.arange(4, dtype=np.float32), np.full(4, 2., np.float32), None))
    result = deployment.Route(dict(family="CURRENT", run_id="synthetic"), data, tmp_path,
                              device="cpu", foundation_pool=pool(data))
    if mode == "RAW":
        result.family = "RAW"
    elif mode != "CURRENT":
        base = ForecastAdapter("B2", data["L"], data["H"], 4, 2, np.zeros(4), np.eye(4))
        result.model = models.convert_model(base, mode, schedule=schedule).eval()
        result.family, result.latent_k = "B2", 2
    return result


def test_actual_b_forward_uses_two_lengths_one_pool_and_original_h(tmp_path, monkeypatch):
    actual = route(tmp_path, monkeypatch)
    physical, observed = costs.windows(actual.task, np.array([512, 524], np.int64))
    prediction = actual.predict_physical(physical, observed)
    assert prediction.shape == (2, 12, 4)
    assert np.isfinite(prediction).all()
    measured = costs.check_work(actual, 2)
    assert [v["input_shape"] for v in measured["F_invocations"]] == [[2, 512, 1], [2, 256, 1]]
    assert [v["used_horizon"] for v in measured["F_invocations"]] == [12, 6]
    assert [v["wrapper_horizon"] for v in measured["F_invocations"]] == [12, 12]
    assert [v["actual_head_horizon"] for v in measured["F_invocations"]] == [64, 64]
    assert actual.pool.instances_created == 1
    assert actual.pool.foundation.horizon == 12
    assert measured["F_calls"] == 2
    actual.close()


def test_a_cost_only_schedule_same_weights_output_and_different_calls(tmp_path, monkeypatch):
    one = route(tmp_path, monkeypatch, "A")
    two = route(tmp_path, monkeypatch, "A", "two_call")
    two.model.load_state_dict(one.model.state_dict())
    before = {key: value.clone() for key, value in one.model.state_dict().items()}
    physical, observed = costs.windows(one.task, np.array([512], np.int64))
    p_one = one.predict_physical(physical, observed)
    p_two = two.predict_physical(physical, observed)
    np.testing.assert_allclose(p_one, p_two, atol=common.ATOL, rtol=common.RTOL)
    assert costs.check_work(one, 1)["F_calls"] == 1
    work = costs.check_work(two, 1)
    assert work["F_calls"] == 2
    assert [v["requested_context"] for v in work["F_invocations"]] == [512, 512]
    for key, value in before.items():
        assert torch.equal(value, two.model.state_dict()[key])
    one.close()
    two.close()


@pytest.mark.parametrize("alpha,needed", [(0., ["G"]), (1., ["B"]), (.5, ["G", "B"])])
def test_alpha_loads_and_calls_only_needed_parents(alpha, needed, tmp_path, monkeypatch):
    data = task()
    monkeypatch.setattr(deployment.legacy, "fit_statistics",
        lambda t, end: (np.zeros(4, np.float32), np.ones(4, np.float32), None))
    loaded, children = [], []
    def child(ctx, task_value, spec, *, device, foundation_pool):
        loaded.append(spec["name"])
        parent = deployment.Route(dict(family="CURRENT"), task_value, tmp_path,
                                  device=device, foundation_pool=foundation_pool)
        children.append(parent)
        return parent
    monkeypatch.setattr(deployment, "make_route", child)
    combined = deployment.CombinedRoute(dict(family="COMBINED", operation="alpha", alpha=alpha,
        g_route={"name": "G"}, b_route={"name": "B"}), data, tmp_path,
        ctx={}, device="cpu", foundation_pool=pool(data))
    assert loaded == needed
    physical, observed = costs.windows(data, np.array([512], np.int64))
    prediction = combined.predict_physical(physical, observed)
    np.testing.assert_array_equal(prediction, np.repeat(physical[:, -1:].astype(np.float64), 12, axis=1))
    assert costs.check_work(combined, 1)["F_calls"] == 0
    assert combined.pool.instances_created == 0
    combined.close()


def test_past_control_never_loads_b_and_cp0_equals_g(tmp_path, monkeypatch):
    data = task()
    monkeypatch.setattr(deployment.legacy, "fit_statistics",
        lambda t, end: (np.zeros(4, np.float32), np.ones(4, np.float32), None))
    head = ConditionalHead(data["L"], data["H"], hidden=32)
    path = tmp_path/"head.pt"
    torch.save(dict(state=head.state_dict(), head_config=dict(length=512, horizon=12, hidden=32)), path)
    loaded = []
    def child(ctx, task_value, spec, *, device, foundation_pool):
        loaded.append(spec["family"])
        assert spec["family"] == "DIRECT_G"
        return deployment.Route(dict(family="CURRENT"), task_value, tmp_path,
                                device=device, foundation_pool=foundation_pool)
    monkeypatch.setattr(deployment, "make_route", child)
    spec = dict(family="COMBINED", operation="past_control", g_route={"family": "DIRECT_G"},
                b_route={"family": "B2", "invalid_unused_artifact": True},
                head_state_path=path.name, head_state_sha256=common.sha(path))
    combined = deployment.CombinedRoute(spec, data, tmp_path, ctx={}, device="cpu", foundation_pool=pool(data))
    physical, observed = costs.windows(data, np.array([512], np.int64))
    prediction = combined.predict_physical(physical, observed)
    np.testing.assert_array_equal(prediction, np.repeat(physical[:, -1:].astype(np.float64), 12, axis=1))
    assert loaded == ["DIRECT_G"]
    assert costs.check_work(combined, 1)["F_calls"] == 0
    assert combined.pool.foundation is None
    combined.close()


def test_head_path_hash_and_dimensions_are_enforced(tmp_path, monkeypatch):
    data = task()
    monkeypatch.setattr(deployment.legacy, "fit_statistics",
        lambda t, end: (np.zeros(4, np.float32), np.ones(4, np.float32), None))
    monkeypatch.setattr(deployment, "make_route", lambda *a, **kw: None)
    path = tmp_path/"head.pt"
    torch.save(dict(state=ConditionalHead(512, 12).state_dict(),
                    head_config=dict(length=512, horizon=12, hidden=31)), path)
    spec = dict(family="COMBINED", operation="past_control", g_route={},
                head_state_path=path.name, head_state_sha256=common.sha(path))
    with pytest.raises(ValueError, match="hidden32"):
        deployment.CombinedRoute(spec, data, tmp_path, ctx={}, device="cpu", foundation_pool=pool(data))
    spec["head_state_sha256"] = "bad"
    with pytest.raises(ValueError, match="SHA256"):
        deployment.CombinedRoute(spec, data, tmp_path, ctx={}, device="cpu", foundation_pool=pool(data))


def test_origin_schedule_is_matched_and_cycles_without_changing_order():
    origins = np.array([100, 112, 124], np.int64)
    b1 = costs.timing_origin_bundles(origins, 1)
    b8 = costs.timing_origin_bundles(origins, 8)
    assert [int(v[0]) for v in b1] == [100, 112, 124, 100, 112, 124, 100, 112]
    np.testing.assert_array_equal(np.concatenate(b1), b8[0])


def test_latency_summary_uses_only_timed_and_separates_batch_origin():
    trials = [dict(phase="warmup", whole_seconds=100.)]
    trials += [dict(phase="timed", whole_seconds=value/1000) for value in range(1, 21)]
    summary = costs.latency_statistics(trials, 8)
    assert summary["p50_batch_ms"] == 10.5
    assert summary["IQR_batch_ms"] == 9.5
    assert summary["p95_batch_ms"] == pytest.approx(19.05)
    assert summary["p50_ms_per_origin"] == 10.5/8
    assert summary["p95_ms_per_origin"] == pytest.approx(19.05/8)


def test_call_gate_rejects_padding_or_hidden_second_call(tmp_path, monkeypatch):
    actual = route(tmp_path, monkeypatch)
    physical, observed = costs.windows(actual.task, np.array([512], np.int64))
    actual.predict_physical(physical, observed)
    actual.last_receipt["F_invocations"][1]["requested_context"] = 512
    with pytest.raises(ValueError, match="branch length"):
        costs.check_work(actual, 1)
    actual.last_receipt["F_calls"] = 1
    with pytest.raises(ValueError, match="physical work"):
        costs.check_work(actual, 1)
    actual.close()


def request(family="B2", batch=1):
    return dict(asset_root="synthetic", dataset="jena", route_spec={"family": family},
                accuracy_run_id="synthetic_accuracy", cost_id="synthetic_cost", batch_size=batch, device="cpu")


def test_ridge_rejects_gpu_before_any_load():
    value = request("RAW_RIDGE")
    value["device"] = "cuda"
    with pytest.raises(ValueError, match="CPU"):
        costs.validate_request(value)
    assert costs.validate_request(request("RAW_RIDGE"))["device"] == "cpu"
    with pytest.raises(ValueError, match="CPU"):
        deployment.make_route({}, {}, {"family": "RAW_RIDGE"}, device="cuda")


class Sampler:
    latest_gpu_mib = None
    def __init__(self, device, interval_seconds):
        assert device == "cpu"
    def start(self):
        pass
    def set_phase(self, phase):
        pass
    def stop(self):
        return dict(CPU_RSS_sampled_peak_MiB=1., process_GPU_sampled_peak_MiB=None,
                    process_GPU_sampling_status="NOT_APPLICABLE_CPU", process_GPU_sampling_reason="CPU")


@pytest.mark.parametrize("batch", [1, 8])
def test_cpu_cost_core_regenerates_full_v_then_measures_actual_forward(batch, tmp_path, monkeypatch):
    actual = route(tmp_path, monkeypatch)
    result = costs.evaluate_inputs(request(batch=batch), {}, actual.task, lambda: actual, sampler_factory=Sampler)
    assert result["status"] == "COMPLETE"
    assert result["parity"]["n_origins"] == 10
    assert result["parity"]["saved_reference"] == "NOT_PROVIDED"
    assert len(result["trials"]) == 30
    assert sum(row["phase"] == "timed" for row in result["trials"]) == 20
    assert all(row["F_calls"] == 2 for row in result["trials"])
    assert result["row"]["foundation_instances_created"] == 1
    assert result["row"]["process_GPU_sampled_peak_MiB"] is None
    assert result["row"]["p50_ms_per_origin"] == result["row"]["p50_batch_ms"]/batch
    assert actual._closed


def test_saved_whole_v_reference_checks_order_mask_values_and_prediction(tmp_path, monkeypatch):
    actual = route(tmp_path, monkeypatch, "A")
    origins = costs.fixed_origins(actual.task)
    inputs = costs.windows(actual.task, origins)
    physical = actual.predict_physical(*inputs)
    rows = origins[:, None] + np.arange(12)[None]
    reference = dict(pred_normalized=((physical-actual.mean)/actual.std).astype(np.float32),
        target_normalized=((actual.task["physical"][rows]-actual.mean)/actual.std).astype(np.float32),
        origins=origins.copy(), observed=actual.task["observed"][rows].copy(), channel_ids=np.array(actual.task["columns"]))
    result = costs.whole_v_parity(actual, actual.task, origins, inputs, reference, 8)
    assert result["saved_reference"] == "PASS"
    for key in ("origins", "observed", "pred_normalized"):
        changed = copy.deepcopy(reference)
        if key == "origins": changed[key] = changed[key][::-1]
        elif key == "observed": changed[key][0, 0, 0] = False
        else: changed[key][0, 0, 0] += 1
        with pytest.raises(ValueError):
            costs.whole_v_parity(actual, actual.task, origins, inputs, changed, 8)
    actual.close()


def test_resident_counts_shared_f_once(tmp_path, monkeypatch):
    actual = route(tmp_path, monkeypatch)
    actual.pool.get(12)
    state = deployment.resident_state(actual)
    assert state["trainable_parameters"] == sum(p.numel() for p in actual.model.parameters())
    assert state["resident_parameters"] == state["trainable_parameters"] + 2
    assert state["resident_buffer_elements"] == 4
    assert state["resident_weights_gpu_MiB"] == 0
    actual.close()


def test_raw_b8_preserves_all_21_channels_with_two_row_chunks(tmp_path, monkeypatch):
    data = task()
    data["columns"] = ["c"+str(i) for i in range(21)]
    data["physical"] = np.arange(700*21, dtype=np.float32).reshape(700, 21)/1000
    data["observed"] = np.ones((700, 21), bool)
    monkeypatch.setattr(deployment.legacy, "fit_statistics",
        lambda t, end: (np.zeros(21, np.float32), np.ones(21, np.float32), None))
    actual = deployment.Route(dict(family="CURRENT", row_microbatch=128), data, tmp_path,
                              device="cpu", foundation_pool=pool(data))
    actual.family = "RAW"
    physical, observed = costs.windows(data, costs.fixed_origins(data)[:8])
    prediction = actual.predict_physical(physical, observed)
    measured = costs.check_work(actual, 8)
    assert prediction.shape == (8, 12, 21)
    assert measured["F_calls"] == 2
    assert measured["F_physical_streams"] == 168
    assert measured["F_invocations"][0]["input_shape"] == [8, 512, 21]
    actual.close()


def test_native_numpy_ridge_executes_whole_cpu_path_without_f(tmp_path, monkeypatch):
    actual = route(tmp_path, monkeypatch, "CURRENT")
    actual.family, actual.latent_k = "RAW_RIDGE", 0
    coefficients = np.zeros((4, 33, 12), np.float64)
    coefficients[:, -1] = .25
    actual.ridge_artifact = dict(coefficients=coefficients, mean=actual.mean, std=actual.std)
    physical, observed = costs.windows(actual.task, np.array([512], np.int64))
    prediction = actual.predict_physical(physical, observed)
    expected = np.repeat(physical[:, -1:].astype(np.float64), 12, axis=1)+.25*actual.std
    np.testing.assert_allclose(prediction, expected, atol=2e-7)
    assert prediction.dtype == np.float64
    assert costs.check_work(actual, 1)["F_calls"] == 0
    assert actual.pool.foundation is None
    result = costs.evaluate_inputs(request("RAW_RIDGE"), {}, actual.task, lambda: actual, sampler_factory=Sampler)
    assert result["row"]["comparison_scope"] == "CPU_RIDGE_CONTEXT"
    assert result["row"]["memory_layer"] == "cpu_RSS"
    assert result["row"]["peak_allocated_MiB"] is None
    assert result["row"]["foundation_instances_created"] == 0
