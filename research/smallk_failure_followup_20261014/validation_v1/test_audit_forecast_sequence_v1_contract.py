from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parent / "audit_forecast_sequence_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("audit_forecast_sequence_v1_contract", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Checks:
    def __init__(self):
        self.rows = []

    def __call__(self, name, passed, details=None):
        self.rows.append((name, bool(passed), details))

    def passed(self, name):
        return any(row_name == name and passed for row_name, passed, _ in self.rows)


def test_hash_algorithms_keep_old_and_bi_contracts_distinct():
    audit = load_module()
    state = {
        "codec.decode.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "codec.decode.bias": np.array([0.25, -0.5], dtype=np.float32),
    }
    head = {"output.bias": np.array([0.0, 1.0], dtype=np.float32)}

    old_hash = audit.old_state_hash(state)
    bi_hash = audit.bi_state_hash(state)
    combined = audit.combined_bi_hash(state, head)

    assert old_hash != bi_hash
    assert combined == audit.bi_state_hash({
        "codec.codec.decode.weight": state["codec.decode.weight"],
        "codec.codec.decode.bias": state["codec.decode.bias"],
        "head.output.bias": head["output.bias"],
    })


def test_static_rf1_reconstruction_uses_decoder_anchor_and_zero_head():
    audit = load_module()
    codec = {
        "codec.decode.weight": torch.tensor([[1.0, 0.0], [0.5, -1.0]], dtype=torch.float32),
        "codec.decode.bias": torch.tensor([0.25, -0.25], dtype=torch.float32),
        "residual.0.weight": torch.tensor([[0.2, -0.1], [0.0, 0.3], [-0.4, 0.5]], dtype=torch.float32),
        "residual.0.bias": torch.tensor([0.1, -0.2, 0.0], dtype=torch.float32),
        "residual.2.weight": torch.tensor([[0.1, 0.0, -0.2], [0.05, -0.15, 0.2]], dtype=torch.float32),
        "residual.2.bias": torch.tensor([0.01, -0.02], dtype=torch.float32),
    }
    head = {
        "hidden.weight": torch.tensor([[1.0, -1.0], [0.5, 0.25], [-0.25, 0.75]], dtype=torch.float32),
        "hidden.bias": torch.tensor([0.0, 0.1, -0.2], dtype=torch.float32),
        "output.weight": torch.zeros((2, 3), dtype=torch.float32),
        "output.bias": torch.zeros(2, dtype=torch.float32),
    }
    arrays = {
        "raw_last": np.array([[[10.0, -2.0]]], dtype=np.float32),
        "past_z_last_two": np.array([[[0.0, 0.0], [1.0, 2.0]]], dtype=np.float32),
        "absolute_future_z": np.array([[[2.0, 0.0], [3.0, -1.0]]], dtype=np.float32),
    }

    base, residual, prediction = audit.reconstruct_saved_trace(torch, codec, head, "STATIC_RF1", arrays)
    future_decoded = audit.codec_decode(torch, codec, arrays["absolute_future_z"]).detach().numpy()
    last_decoded = audit.codec_decode(torch, codec, arrays["past_z_last_two"][:, -1:]).detach().numpy()
    expected_base = arrays["raw_last"] + future_decoded - last_decoded

    np.testing.assert_allclose(base, expected_base.astype(np.float32), atol=1e-6)
    np.testing.assert_array_equal(residual, np.zeros_like(residual))
    np.testing.assert_allclose(prediction, expected_base.astype(np.float32), atol=1e-6)


def test_temporal_rf3_residual_uses_last_two_past_latents_and_future_window():
    audit = load_module()
    head = {
        "hidden.weight": torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) / 10.0,
        "hidden.bias": torch.tensor([0.1, -0.2], dtype=torch.float32),
        "output.weight": torch.tensor([[[0.5], [-0.25]]], dtype=torch.float32),
        "output.bias": torch.tensor([0.05], dtype=torch.float32),
    }
    past = np.array([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32)
    future = np.array([[[4.0, 5.0], [6.0, 7.0]]], dtype=np.float32)

    residual = audit.head_residual(torch, "TEMPORAL_RF3", head, past, future).detach().numpy()
    repeated_last = np.repeat(past[:, -1:], future.shape[1], axis=1)
    response_future = torch.nn.functional.conv1d(
        torch.nn.functional.gelu(torch.nn.functional.conv1d(
            torch.tensor(np.concatenate([past, future], axis=1).transpose(0, 2, 1)),
            head["hidden.weight"], head["hidden.bias"])),
        head["output.weight"], head["output.bias"]).transpose(1, 2).detach().numpy()
    response_last = torch.nn.functional.conv1d(
        torch.nn.functional.gelu(torch.nn.functional.conv1d(
            torch.tensor(np.concatenate([past, repeated_last], axis=1).transpose(0, 2, 1)),
            head["hidden.weight"], head["hidden.bias"])),
        head["output.weight"], head["output.bias"]).transpose(1, 2).detach().numpy()

    np.testing.assert_allclose(residual, response_future - response_last, atol=1e-6)
    assert residual.shape == (1, 2, 1)


def test_score_all_uses_channel_macro_mask_and_raw_teacher():
    audit = load_module()
    prediction = np.array([[[1.0, 4.0], [3.0, 8.0]]], dtype=np.float32)
    target = np.array([[[0.0, 1.0], [1.0, 2.0]]], dtype=np.float32)
    raw = np.array([[[1.0, 2.0], [1.0, 10.0]]], dtype=np.float32)
    observed = np.array([[[True, True], [False, True]]])
    values = {"target": target, "observed": observed, "raw_forecast": raw}
    scores = audit.score_all(prediction, values, np.array([2.0, 0.5], dtype=np.float32))

    # channel0 has one observed squared error 1; channel1 has errors 3 and 6, squared 9 and 36.
    assert scores["normalized_actual_Y"]["mse"] == pytest.approx((1.0 + 22.5) / 2.0)
    # RAW teacher comparison uses prediction - raw under the same observed mask.
    assert scores["normalized_RAW_teacher"]["mae"] == pytest.approx((0.0 + ((2.0 + 2.0) / 2.0)) / 2.0)
    # Physical score scales residuals by std before reducing per channel.
    assert scores["physical_actual_Y"]["mae"] == pytest.approx((2.0 + ((1.5 + 3.0) / 2.0)) / 2.0)


def test_queue_schema_expands_eight_producer_refs(tmp_path):
    audit = load_module()
    assert "forecast_sequence_v1" in str(audit.DEFAULT_QUEUE)
    assert audit.splits_for("robin") == ("fit", "tune", "val")
    assert audit.splits_for("beijing_aotizhongxin") == ("fit", "tune", "eval")
    conditions = []
    for dataset in audit.DATASETS:
        for seed in audit.SEEDS:
            for head in audit.HEADS:
                conditions.append({
                    "tag": f"sequence_{dataset}_{audit.HEAD_TAG[head]}_s{seed}_v1",
                    "dataset": dataset,
                    "seed": seed,
                    "head": head,
                    "status": "COMPLETE",
                    "selected_new_update": 100,
                    "producer_result": {"path": str(tmp_path / f"{dataset}_{seed}_{head}.json"), "sha256": "abc"},
                })
    queue = tmp_path / "FORECAST_SEQUENCE_QUEUE_v1.json"
    queue.write_text(json.dumps({"status": "COMPLETE", "conditions": conditions}), encoding="utf-8")
    checks = Checks()

    refs = audit.producer_refs_from_inputs([queue], checks)

    assert len(refs) == 8
    assert checks.passed("producer_ref_count")
    assert {ref["_queue_condition"]["head"] for ref in refs} == set(audit.HEADS)


def test_prefixed_checkpoint_trace_view_uses_fit_tune_fields_only():
    audit = load_module()
    arrays = {
        "fit_prediction": np.ones((1, 2, 1), dtype=np.float32),
        "fit_past_z_last_two": np.ones((1, 2, 1), dtype=np.float32) * 2,
        "fit_absolute_future_z": np.ones((1, 2, 1), dtype=np.float32) * 3,
        "fit_base_MAIN_style_prediction": np.ones((1, 2, 1), dtype=np.float32) * 4,
        "fit_head_residual": np.ones((1, 2, 1), dtype=np.float32) * 5,
        "fit_raw_last": np.ones((1, 1, 1), dtype=np.float32) * 6,
        "tune_prediction": np.zeros((1, 2, 1), dtype=np.float32),
        "eval_prediction": np.ones((1, 2, 1), dtype=np.float32) * 99,
    }

    fit = audit.prefixed_trace(arrays, "fit")

    assert set(fit) == set(audit.TRACE_FIELDS)
    np.testing.assert_array_equal(fit["prediction"], arrays["fit_prediction"])
    np.testing.assert_array_equal(fit["raw_last"], arrays["fit_raw_last"])


def test_parameter_delta_groups_separates_encoder_decoder_and_head():
    audit = load_module()
    initial_codec = {
        "project.weight": torch.tensor([1.0]),
        "codec.decode.weight": torch.tensor([2.0]),
        "residual.0.weight": torch.tensor([0.0]),
        "codec.decode.bias": torch.tensor([0.0]),
        "fixed_basis": torch.tensor([100.0]),
    }
    final_codec = {
        "project.weight": torch.tensor([1.5]),
        "codec.decode.weight": torch.tensor([2.0]),
        "residual.0.weight": torch.tensor([0.5]),
        "codec.decode.bias": torch.tensor([0.0]),
        "fixed_basis": torch.tensor([200.0]),
    }
    initial_head = {"output.weight": torch.tensor([0.0])}
    final_head = {"output.weight": torch.tensor([1.0])}

    groups = audit.parameter_delta_groups(initial_codec, final_codec, initial_head, final_head)

    assert groups["E"]["changed"] == 1
    assert groups["D"]["changed"] == 1
    assert groups["Head"]["changed"] == 1

