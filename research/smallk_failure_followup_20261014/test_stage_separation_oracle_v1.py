import sys
from pathlib import Path

import numpy as np
import pytest
import torch


MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

import stage_separation_oracle_v1 as oracle  # noqa: E402


class CausalRF2Encoder:
    """Tiny causal encoder used only for source-level contract tests."""

    def encode(self, x):
        previous = torch.zeros_like(x)
        previous[:, 1:] = x[:, :-1]
        return x + 0.25 * previous


class PoisonFutureArray:
    def __array__(self, dtype=None):
        raise AssertionError("future label/mask must not be read by past-only deploy prep")


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float32)


def no_grad():
    return torch.no_grad()


def check():
    return None


def test_causal_future_fill_uses_filled_past_last_then_previous_observed_future():
    history = np.array([[[1.0, 10.0], [2.0, 20.0]]], dtype=np.float32)
    history_observed = np.ones_like(history, dtype=np.bool_)
    target = np.array([[[100.0, np.nan], [101.0, 201.0], [np.nan, np.nan]]], dtype=np.float32)
    observed = np.array([[[True, False], [True, True], [False, False]]], dtype=np.bool_)

    future, source, complete = oracle.causal_fill_future_from_past(
        history, history_observed, target, observed
    )

    np.testing.assert_array_equal(
        future,
        np.array([[[100.0, 20.0], [101.0, 201.0], [101.0, 201.0]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        source,
        np.array(
            [
                [
                    [oracle.FILL_OBSERVED, oracle.FILL_CARRY_PAST_LAST],
                    [oracle.FILL_OBSERVED, oracle.FILL_OBSERVED],
                    [oracle.FILL_CARRY_PREVIOUS_FUTURE, oracle.FILL_CARRY_PREVIOUS_FUTURE],
                ]
            ],
            dtype=np.uint8,
        ),
    )
    np.testing.assert_array_equal(complete, np.array([[False, False, False]], dtype=np.bool_))


def test_complete_future_prefix_mask_requires_all_channels_observed_so_far():
    history = np.zeros((2, 4, 2), dtype=np.float32)
    history_observed = np.ones_like(history, dtype=np.bool_)
    target = np.zeros((2, 3, 2), dtype=np.float32)
    observed = np.array(
        [
            [[True, True], [True, True], [False, True]],
            [[True, False], [True, True], [True, True]],
        ],
        dtype=np.bool_,
    )

    context = oracle.build_oracle_encoder_context(history, history_observed, target, observed)

    np.testing.assert_array_equal(
        context.complete_future_prefix_mask,
        np.array([[True, True, False], [False, False, False]], dtype=np.bool_),
    )
    teacher = oracle.complete_future_teacher_mask(context.complete_future_prefix_mask, latent_dim=3)
    assert teacher.shape == (2, 3, 3)
    np.testing.assert_array_equal(teacher[:, :, 0], context.complete_future_prefix_mask)


def test_oracle_context_is_560_and_keeps_observed_mask_separate_from_filled_values():
    history = np.zeros((1, 512, 2), dtype=np.float32)
    history[:, -1, :] = np.array([5.0, -5.0], dtype=np.float32)
    history_observed = np.ones_like(history, dtype=np.bool_)
    target = np.zeros((1, 48, 2), dtype=np.float32)
    target[:, 0, 0] = 9.0
    observed = np.zeros_like(target, dtype=np.bool_)
    observed[:, 0, 0] = True

    context = oracle.build_oracle_encoder_context(history, history_observed, target, observed)

    assert context.encoder_context.shape == (1, 560, 2)
    assert context.encoder_context_observed.shape == (1, 560, 2)
    assert context.encoder_context[0, 512, 0] == np.float32(9.0)
    assert context.encoder_context[0, 512, 1] == np.float32(-5.0)
    assert context.encoder_context_observed[0, 512, 0]
    assert not context.encoder_context_observed[0, 512, 1]


def test_past_only_and_oracle_latent_paths_are_separate_and_prefix_equal_for_causal_encoder():
    codec = CausalRF2Encoder()
    history = np.arange(1 * 6 * 2, dtype=np.float32).reshape(1, 6, 2)
    history_observed = np.ones_like(history, dtype=np.bool_)
    target = np.ones((1, 3, 2), dtype=np.float32)
    observed = np.ones_like(target, dtype=np.bool_)

    past = oracle.extract_past_only_deployment_latents(
        codec,
        history,
        tensor=tensor,
        no_grad=no_grad,
        check=check,
        batch_size=1,
    )
    diagnostic = oracle.extract_true_future_oracle_latents(
        codec,
        history,
        history_observed,
        target,
        observed,
        tensor=tensor,
        no_grad=no_grad,
        check=check,
        batch_size=1,
    )

    report = oracle.verify_past_prefix_parity(
        past["latent_past"], diagnostic["oracle_past_prefix_latent"]
    )
    assert report["pass"], report
    assert diagnostic["oracle_future_latent"].shape == (1, 3, 2)


def test_extract_normalized_oracle_returns_driver_fields_without_renormalizing():
    codec = CausalRF2Encoder()
    history = np.arange(1 * 6 * 2, dtype=np.float32).reshape(1, 6, 2)
    target = np.ones((1, 3, 2), dtype=np.float32)
    observed = np.ones_like(target, dtype=np.bool_)

    extracted = oracle.extract_normalized_oracle(
        codec,
        history,
        target,
        observed,
        tensor=tensor,
        no_grad=no_grad,
        check=check,
        batch_size=1,
    )

    assert extracted["oracle_z"].shape == (1, 3, 2)
    assert extracted["current_past_latent"].shape == (1, 6, 2)
    assert extracted["prefix_check"]["pass"], extracted["prefix_check"]
    assert extracted["complete_future_prefix_counts"] == [1, 1, 1]
    assert extracted["normalization_contract"]["refit_or_renormalized_here"] is False
    assert extracted["first_sample_forward_invariance"]["pass"] is True
    assert extracted["first_sample_forward_invariance"]["context_changed"] is True
    assert extracted["first_sample_forward_invariance"]["perturb_step"] == 2
    assert extracted["first_sample_forward_invariance"]["sample_index"] == 0


def test_extract_normalized_oracle_selects_late_observed_sample_when_first_sample_has_no_future_observation():
    codec = CausalRF2Encoder()
    history = np.zeros((2, 6, 1), dtype=np.float32)
    target = np.ones((2, 3, 1), dtype=np.float32)
    observed = np.array([[[False], [False], [False]], [[False], [True], [True]]], dtype=np.bool_)

    extracted = oracle.extract_normalized_oracle(
        codec,
        history,
        target,
        observed,
        tensor=tensor,
        no_grad=no_grad,
        check=check,
        batch_size=1,
    )

    report = extracted["first_sample_forward_invariance"]
    assert report["pass"] is True, report
    assert report["selected_first_sample"] is False
    assert report["sample_index"] == 1
    assert report["perturb_step"] == 2
    assert report["selected_step_reason"] == "latest_future_step_with_observed_channel"


def test_extract_normalized_oracle_reports_no_pass_claim_when_no_future_is_observed():
    codec = CausalRF2Encoder()
    history = np.zeros((1, 6, 1), dtype=np.float32)
    target = np.ones((1, 3, 1), dtype=np.float32)
    observed = np.zeros_like(target, dtype=np.bool_)

    extracted = oracle.extract_normalized_oracle(
        codec,
        history,
        target,
        observed,
        tensor=tensor,
        no_grad=no_grad,
        check=check,
        batch_size=1,
    )

    assert extracted["first_sample_forward_invariance"]["status"] == "NOT_TESTED_NO_OBSERVED_FUTURE"
    assert extracted["first_sample_forward_invariance"]["pass"] is None


def test_missing_future_perturbation_is_ignored_and_observed_later_perturbation_preserves_earlier_prefix():
    codec = CausalRF2Encoder()
    history = np.zeros((1, 5, 1), dtype=np.float32)
    history[:, -1, :] = 2.0
    history_observed = np.ones_like(history, dtype=np.bool_)
    target = np.array([[[3.0], [4.0], [5.0]]], dtype=np.float32)
    observed_missing_middle = np.array([[[True], [False], [True]]], dtype=np.bool_)

    missing_report = oracle.verify_future_perturb_prefix_invariance(
        codec,
        history,
        history_observed,
        target,
        observed_missing_middle,
        perturb_step=1,
        perturb_delta=100.0,
        tensor=tensor,
        no_grad=no_grad,
        check=check,
        batch_size=1,
    )
    assert missing_report["filled_context_all"]["pass"], missing_report
    assert missing_report["status"] == "NOT_TESTED_PERTURBATION_DID_NOT_CHANGE_FILLED_CONTEXT"
    assert missing_report["pass"] is None

    observed_all = np.ones_like(target, dtype=np.bool_)
    later_report = oracle.verify_future_perturb_prefix_invariance(
        codec,
        history,
        history_observed,
        target,
        observed_all,
        perturb_step=2,
        perturb_delta=100.0,
        tensor=tensor,
        no_grad=no_grad,
        check=check,
        batch_size=1,
    )
    assert later_report["pass"] is True
    assert later_report["context_changed"] is True
    assert later_report["context_prefix"]["pass"], later_report
    assert later_report["latent_prefix"]["pass"], later_report
    assert not later_report["filled_context_all"]["pass"], later_report
    assert "max_abs" in later_report["latent_suffix_difference"]


def test_validate_split_payload_rejects_refit_or_misaligned_stats_and_raw_last():
    history = np.zeros((1, 512, 2), dtype=np.float32)
    history_observed = np.ones_like(history, dtype=np.bool_)
    payload = {
        "history": history,
        "history_observed": history_observed,
        "target": np.zeros((1, 48, 2), dtype=np.float32),
        "observed": np.ones((1, 48, 2), dtype=np.bool_),
        "origins": np.array([7], dtype=np.int64),
        "raw_last": np.ones((1, 2), dtype=np.float32),
        "mean": np.zeros(2, dtype=np.float32),
        "std": np.ones(2, dtype=np.float32),
    }

    with pytest.raises(oracle.StageSeparationContractError, match="RAW_LAST"):
        oracle.validate_split_payload(payload)


def test_prepare_past_only_deployment_inputs_accepts_payload_without_future_labels():
    history = np.zeros((1, 512, 2), dtype=np.float32)
    history[:, -1] = np.array([1.0, 2.0], dtype=np.float32)
    payload = {
        "history": history,
        "history_observed": np.ones_like(history, dtype=np.bool_),
        "origins": np.array([7], dtype=np.int64),
        "raw_last": history[:, -1].copy(),
        "mean": np.zeros(2, dtype=np.float32),
        "std": np.ones(2, dtype=np.float32),
    }

    prepared = oracle.prepare_past_only_deployment_inputs(payload)

    np.testing.assert_array_equal(prepared.history, history)
    np.testing.assert_array_equal(prepared.raw_last, history[:, -1])


def test_prepare_past_only_deployment_inputs_does_not_touch_poison_future_arrays():
    history = np.zeros((1, 512, 1), dtype=np.float32)
    payload = {
        "history": history,
        "history_observed": np.ones_like(history, dtype=np.bool_),
        "origins": np.array([3], dtype=np.int64),
        "raw_last": history[:, -1].copy(),
        "mean": np.zeros(1, dtype=np.float32),
        "std": np.ones(1, dtype=np.float32),
        "target": PoisonFutureArray(),
        "observed": PoisonFutureArray(),
    }

    prepared = oracle.prepare_past_only_deployment_inputs(payload)

    assert prepared.history.shape == (1, 512, 1)


def test_full_split_validation_still_requires_future_labels_by_default():
    history = np.zeros((1, 512, 1), dtype=np.float32)
    payload = {
        "history": history,
        "history_observed": np.ones_like(history, dtype=np.bool_),
        "origins": np.array([3], dtype=np.int64),
        "raw_last": history[:, -1].copy(),
        "mean": np.zeros(1, dtype=np.float32),
        "std": np.ones(1, dtype=np.float32),
    }

    with pytest.raises(oracle.StageSeparationContractError, match="target"):
        oracle.validate_split_payload(payload)


def test_d2_cached_latent_schema_matches_source_extraction_keys():
    payload = {
        "latent_past": np.zeros((2, 512, 3), dtype=np.float32),
        "z_future": np.zeros((2, 48, 3), dtype=np.float32),
        "z_last": np.zeros((2, 1, 3), dtype=np.float32),
        "source_gpu_prediction": np.zeros((2, 48, 4), dtype=np.float32),
        "original_prediction": np.zeros((2, 48, 4), dtype=np.float32),
        "raw_forecast": np.zeros((2, 48, 4), dtype=np.float32),
        "target": np.zeros((2, 48, 4), dtype=np.float32),
        "observed": np.ones((2, 48, 4), dtype=np.bool_),
        "origins": np.array([10, 20], dtype=np.int64),
        "raw_last": np.zeros((2, 4), dtype=np.float32),
        "history_observed": np.ones((2, 512, 4), dtype=np.bool_),
    }

    record = oracle.validate_d2_cached_latent_payload(payload, latent_dim=3, channels=4)

    assert record == {"windows": 2, "history_length": 512, "horizon": 48, "channels": 4, "latent_dim": 3}


def test_protected_test_path_guard_is_side_effect_free_and_path_based():
    with pytest.raises(oracle.ProtectedTestPathError):
        oracle.assert_no_protected_test_path({"path": "E:/old_assets/TEST/eval_target.npz"}, role="future")

    oracle.assert_no_protected_test_path(
        {"path": "E:/CODING/proj/hierarchical-tsfm-peft/research/smallk_failure_followup_20261014/test_stage_separation_oracle_v1.py"},
        role="source_test_file",
    )


def test_file_record_contract_checks_sha_without_opening_path():
    record = oracle.validate_file_record_contract(
        {
            "path": "E:/old_assets/source/engine.py",
            "sha256": "a" * 64,
        },
        role="OLD_ENGINE_SOURCE",
    )

    assert record["role"] == "OLD_ENGINE_SOURCE"
    assert record["sha256"] == "a" * 64

    with pytest.raises(oracle.StageSeparationContractError, match="SHA256"):
        oracle.validate_file_record_contract(
            {"path": "E:/old_assets/source/engine.py", "sha256": "not-a-sha"},
            role="OLD_ENGINE_SOURCE",
        )
