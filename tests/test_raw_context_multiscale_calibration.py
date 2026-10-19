"""CPU-only calibration contracts; no source arrays, parents or GPU are opened."""
import importlib
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def calibration():
    try:
        return importlib.import_module("research.raw_context_multiscale_20261018_v1.calibration")
    except ModuleNotFoundError as error:
        pytest.fail(f"M_cal-only calibration implementation is missing: {error}")


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def toy():
    task = {"L": 8, "H": 4, "columns": ["c0", "c1"],
            "split_ranges": {"M_cal": [20, 72]}}
    origins = np.arange(20, 69, dtype=np.int64)
    rng = np.random.default_rng(481)
    past = rng.normal(size=(len(origins), 8, 2)).astype(np.float32)
    g = np.repeat(past[:, -1:], 4, axis=1)
    b = (g + .3).astype(np.float32)
    y = (g + .2).astype(np.float32)
    mask = np.ones(y.shape, dtype=bool)
    mask[::3, :, 1] = False
    return past, g, b, y, mask, origins, task


@pytest.mark.parametrize("bounds,horizon,mcut,nfit,nselect,discard", [
    ((12579, 14675), 48, 13836, 1210, 792, 47),
    ((10713, 12499), 12, 11784, 1060, 704, 11),
])
def test_dense_calibration_split_has_disjoint_target_rows(bounds, horizon, mcut, nfit, nselect, discard):
    split = calibration().calibration_split({"L": 512, "H": horizon,
                                           "split_ranges": {"M_cal": bounds}})
    assert split["mcut"] == mcut
    assert len(split["fit_indices"]) == nfit
    assert len(split["select_indices"]) == nselect
    assert len(split["discarded_indices"]) == discard
    origins = split["origins"]
    fit_targets = {int(o + h) for o in origins[split["fit_indices"]] for h in range(horizon)}
    select_targets = {int(o + h) for o in origins[split["select_indices"]] for h in range(horizon)}
    assert not fit_targets.intersection(select_targets)
    np.testing.assert_array_equal(origins, np.arange(bounds[0], bounds[1] - horizon + 1))


def test_alpha_uses_equal_channel_weights_and_refits_full_calibration():
    g = np.zeros((6, 1, 2))
    b = np.ones_like(g)
    y = np.zeros_like(g)
    y[:, :, 1] = 1
    mask = np.ones(g.shape, dtype=bool)
    mask[1:4, :, 1] = False
    result = calibration().fit_alpha_diagnostic(g, b, y, mask, np.arange(4), np.arange(4, 6))
    assert result["fit_alpha"]["alpha"] == .5
    assert result["refit_alpha"]["alpha"] == .5
    assert result["fit_alpha"]["alpha"] != pytest.approx(1 / 5)
    assert [row["label"] for row in result["rows"]] == ["alpha0", "alpha05", "alpha1", "fitted"]
    assert all("M_fit_mse" in row and "M_select_mse" in row for row in result["rows"])
    json.dumps(result, allow_nan=False)


def test_alpha_zero_denominator_records_zero_and_reason():
    g = np.zeros((4, 2, 1))
    result = calibration().fit_alpha_diagnostic(g, g.copy(), g + 1, np.ones(g.shape, bool), [0], [2, 3])
    assert result["fit_alpha"]["alpha"] == 0
    assert result["fit_alpha"]["zero_denominator"] is True
    assert result["refit_alpha"]["zero_denominator"] is True


def test_alpha_split_must_be_disjoint_and_valid():
    g = np.zeros((4, 2, 1))
    with pytest.raises(ValueError, match="disjoint"):
        calibration().fit_alpha_diagnostic(g, g, g, np.ones(g.shape, bool), [0, 1], [1, 2])


def test_alpha_rejects_disjoint_origins_whose_targets_overlap():
    g = np.zeros((4, 2, 1))
    with pytest.raises(ValueError, match="targets overlap"):
        calibration().fit_alpha_diagnostic(g, g, g, np.ones(g.shape, bool), [0, 1], [2, 3])


def test_initial_head_output_equals_g_exactly_for_conditional_and_control():
    past, g, b, y, mask, origins, task = toy()
    initial_hashes = []
    for control in (False, True):
        state, receipt = calibration().head_refit(past, g, b, y, mask, origins, task, 92601, 0, control=control)
        prediction = calibration().predict_head(state, past, g, b, control=control)
        np.testing.assert_array_equal(prediction, g)
        assert receipt["updates"] == receipt["exposures"] == 0
        assert all(value.device.type == "cpu" for value in state.values())
        initial_hashes.append(receipt["initialization_sha256"])
    assert len(set(initial_hashes)) == 1


def test_heads_share_initialization_sample_order_batch_opportunities():
    values = toy()
    results = [calibration().head_search(*values, 92601, control=control) for control in (False, True)]
    assert results[0]["initialization_sha256"] == results[1]["initialization_sha256"]
    assert results[0]["sampler_sha256"] == results[1]["sampler_sha256"]
    assert [(r["exposures"], r["updates"]) for r in results[0]["curves"]] == [
        (r["exposures"], r["updates"]) for r in results[1]["curves"]]
    assert [r["endpoint_epochs"] for r in results[0]["curves"]] == [0, .25, 1, 3]
    json.dumps(results, allow_nan=False)


@pytest.mark.parametrize("control", [False, True])
def test_checkpoint_sink_receives_all_four_scored_cpu_states_including_cp0(control):
    past, g, b, y, mask, origins, task = toy()
    module = calibration()
    checkpoints = []
    def collect(state, row):
        with torch.random.fork_rng(devices=[]):
            head = module.ConditionalHead(task["L"], task["H"], hidden=32)
        head.load_state_dict(state)
        assert module.module_state_hash(head) == row["checkpoint_state_sha256"]
        assert all(value.device.type == "cpu" for value in state.values())
        checkpoints.append((state, row))
    result = module.head_search(past, g, b, y, mask, origins, task, 92601,
                                control=control, checkpoint_sink=collect)
    assert [row["endpoint_epochs"] for _, row in checkpoints] == [0, .25, 1, 3]
    assert [row for _, row in checkpoints] == result["curves"]
    initial = module.predict_head(checkpoints[0][0], past, g, b, control=control)
    np.testing.assert_array_equal(initial, g)
    select = module.calibration_split(task)["select_indices"]
    for state, row in checkpoints:
        prediction = module.predict_head(state, past[select], g[select], b[select], control=control)
        scores = module.metric_statistics(prediction, y[select], mask[select])
        assert scores["mse"] == row["M_select_mse"]
        assert scores["mae"] == row["M_select_mae"]


def test_checkpoint_sink_observations_cannot_mutate_training_or_selection():
    values = toy()
    module = calibration()
    random_before = torch.get_rng_state().clone()
    expected = module.head_search(*values, 92601)
    final_states = []
    def mutate_copies(state, row):
        if row["endpoint_epochs"] == 3:
            final_states.append({name: value.clone() for name, value in state.items()})
        state["output.bias"].fill_(1e20)
        row["M_select_mse"] = -1e20
    observed = module.head_search(*values, 92601, checkpoint_sink=mutate_copies)
    assert observed == expected
    assert torch.equal(torch.get_rng_state(), random_before)
    assert len(final_states) == 1
    prediction = module.predict_head(final_states[0], values[0], values[1], values[2])
    fit = module.calibration_split(values[-1])["fit_indices"]
    scores = module.metric_statistics(prediction[fit], values[3][fit], values[4][fit])
    assert scores["mse"] == expected["curves"][-1]["M_fit_mse"]


def test_search_and_refit_are_closed_to_validation_data():
    module = calibration()
    for function in (module.head_search, module.head_refit, module.fit_alpha_diagnostic):
        parameters = inspect.signature(function).parameters
        assert not any("v_select" in name.lower() or "validation" in name.lower() for name in parameters)
        assert not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    past, g, b, y, mask, origins, task = toy()
    outside = origins.copy()
    outside[-1] = 72
    with pytest.raises(ValueError, match="dense M_cal"):
        module.head_search(past, g, b, y, mask, outside, task, 1)


def test_select_targets_cannot_change_training_updates_or_fit_curve():
    past, g, b, y, mask, origins, task = toy()
    module = calibration()
    changed = y.copy()
    changed[module.calibration_split(task)["select_indices"]] += 1000
    first = module.head_search(past, g, b, y, mask, origins, task, 92601)
    second = module.head_search(past, g, b, changed, mask, origins, task, 92601)
    assert [r["M_fit_mse"] for r in first["curves"]] == [r["M_fit_mse"] for r in second["curves"]]
    assert first["sampler_sha256"] == second["sampler_sha256"]


def test_cpu_initialization_preserves_callers_random_state():
    values = toy()
    before = torch.get_rng_state().clone()
    calibration().head_refit(*values, 92601, .25)
    assert torch.equal(torch.get_rng_state(), before)


def test_unobserved_nan_targets_do_not_poison_head_gradients():
    past, g, b, y, mask, origins, task = toy()
    y[~mask] = np.nan
    state, receipt = calibration().head_refit(past, g, b, y, mask, origins, task, 92601, .25)
    assert all(torch.isfinite(value).all() for value in state.values())
    assert receipt["actual_epochs"] == pytest.approx(receipt["exposures"] / len(origins))


def test_earliest_checkpoint_wins_exact_tie():
    past, g, b, _, mask, origins, task = toy()
    result = calibration().head_search(past, g, b, g.copy(), mask, origins, task, 7)
    assert result["selected_epoch"] == 0
    assert all(row["M_select_mse"] == 0 for row in result["curves"])


def test_control_never_converts_or_uses_b():
    class ForbiddenParent:
        def __array__(self, *args, **kwargs):
            raise AssertionError("past_control read B2")

    past, g, b, y, mask, origins, task = toy()
    module = calibration()
    state, _ = module.head_refit(past, g, None, y, mask, origins, task, 7, .25, control=True)
    a = module.predict_head(state, past, g, None, control=True)
    z = module.predict_head(state, past, g, ForbiddenParent(), control=True)
    np.testing.assert_array_equal(a, z)
    result = module.head_search(past, g, ForbiddenParent(), y, mask, origins, task, 7, control=True)
    assert result["deployment_parents"] == ["g"]


def test_fresh_refit_does_not_warm_start_or_depend_on_search():
    values = toy()
    module = calibration()
    first, a = module.head_refit(*values, 92601, .25)
    module.head_search(*values, 92601)
    second, b = module.head_refit(*values, 92601, .25)
    assert a["initialization_sha256"] == b["initialization_sha256"]
    assert a["final_state_sha256"] == b["final_state_sha256"]
    assert a["fit_split"] == "M_cal_dense_stride1"
    assert a["exposures"] == int(np.ceil(.25 * len(values[5])))
    for name in first:
        assert torch.equal(first[name], second[name])


def test_microbatch_accumulation_preserves_effective_macro_loss():
    values = toy()
    first, _ = calibration().head_refit(*values, 92601, .25, effective_batch=16, microbatch=1)
    second, _ = calibration().head_refit(*values, 92601, .25, effective_batch=16, microbatch=4)
    for name in first:
        torch.testing.assert_close(first[name], second[name], atol=1e-7, rtol=1e-5)


def test_fp64_statistics_respect_masks_and_reconstruct_channel_macro():
    y = np.array([[[1., np.nan], [2., 4.]], [[3., 5.], [4., np.nan]]])
    mask = np.isfinite(y)
    prediction = np.zeros(y.shape)
    result = calibration().metric_statistics(prediction, y, mask)
    assert result["mse"] == pytest.approx((30 / 4 + 41 / 2) / 2)
    assert result["mae"] == pytest.approx((10 / 4 + 9 / 2) / 2)
    np.testing.assert_array_equal(np.sum(result["horizon_channel_count"], axis=0), result["channel_counts"])
    np.testing.assert_array_equal(np.sum(result["horizon_channel_sse"], axis=0), result["channel_sse"])
    json.dumps(result, allow_nan=False)


def test_parent_scope_requires_frozen_before_calibration():
    task = toy()[-1]
    assert calibration().validate_parent_scope({"parent_fit_end": 20}, task)["parent_fit_end"] == 20
    for metadata in ({}, {"parent_fit_end": 21}, {"parent_fit_end": 19, "parent_selection_end": 21}):
        with pytest.raises(ValueError, match="before M_cal"):
            calibration().validate_parent_scope(metadata, task)


def test_original_bool_mask_and_observed_finite_targets_required():
    past, g, b, y, mask, origins, task = toy()
    with pytest.raises(ValueError, match="bool"):
        calibration().head_search(past, g, b, y, mask.astype(np.int8), origins, task, 1)
    y[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="observed"):
        calibration().head_search(past, g, b, y, mask, origins, task, 1)


@pytest.mark.parametrize("phase", ["search", "refit"])
def test_expired_deadline_refuses_head_work_before_initialization(monkeypatch, phase):
    module = calibration()
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: 5.), raising=False)
    monkeypatch.setattr(module, "_new_head", lambda *args: pytest.fail("Expired head work initialized a model"))
    with pytest.raises(TimeoutError, match="budget reached"):
        if phase == "search":
            module.head_search(*toy(), 92601, deadline=4.)
        else:
            module.head_refit(*toy(), 92601, .25, deadline=4.)


@pytest.mark.parametrize("phase", ["search", "refit"])
def test_head_training_stops_at_expired_microbatch_boundary(monkeypatch, phase):
    module, clock, calls = calibration(), [0.], [0]
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: clock[0]), raising=False)
    original = module.ConditionalHead.forward
    def forward(head, *args, **kwargs):
        result = original(head, *args, **kwargs)
        if torch.is_grad_enabled():
            calls[0] += 1
            clock[0] = 5.
        return result
    monkeypatch.setattr(module.ConditionalHead, "forward", forward)
    with pytest.raises(TimeoutError, match="budget reached"):
        if phase == "search":
            module.head_search(*toy(), 92601, microbatch=1, deadline=4.)
        else:
            module.head_refit(*toy(), 92601, .25, microbatch=1, deadline=4.)
    assert calls[0] == 1


def test_head_search_checks_deadline_between_evaluation_batches(monkeypatch):
    module, clock, calls = calibration(), [0.], [0]
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: clock[0]), raising=False)
    original = module.ConditionalHead.forward
    def forward(head, *args, **kwargs):
        result = original(head, *args, **kwargs)
        if not torch.is_grad_enabled():
            calls[0] += 1
            clock[0] += 1.
        return result
    monkeypatch.setattr(module.ConditionalHead, "forward", forward)
    with pytest.raises(TimeoutError, match="budget reached"):
        module.head_search(*toy(), 92601, deadline=2.5)
    assert calls[0] == 3


def test_available_deadline_does_not_change_scientific_head_results():
    module, values = calibration(), toy()
    assert module.head_search(*values, 92601, deadline=1e12) == module.head_search(*values, 92601)
    state, receipt = module.head_refit(*values, 92601, .25, deadline=1e12)
    reference, expected = module.head_refit(*values, 92601, .25)
    assert receipt == expected
    assert all(torch.equal(state[key], reference[key]) for key in state)
