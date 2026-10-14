from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_stage_native_audit_v2 as audit
import audit_stage_native_core_v2 as core


def test_condition_gate_rejects_duplicate_and_wrong_seed():
    keys = [(d, s) for d in ('robin', 'beijing_aotizhongxin') for s in (92601, 92602)]
    audit.exact_conditions(keys)
    with pytest.raises(ValueError): audit.exact_conditions(keys[:3] + [keys[0]])
    with pytest.raises(ValueError): audit.exact_conditions(keys[:3] + [('beijing_aotizhongxin', 99)])


def test_future_teacher_mask_requires_all_channels_through_prefix():
    mask = np.ones((2, 48, 3), bool)
    mask[0, 4, 1] = False
    teacher = audit.prefix_mask(mask, 2)
    assert teacher.shape == (2, 48, 2)
    assert teacher[0, :4].all() and not teacher[0, 4:].any()
    assert teacher[1].all()


def test_calibration_artifact_hidden_keys_match_original_codec_features():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(2, 48, 2)).astype(np.float32)
    last = rng.normal(size=(2, 1, 2)).astype(np.float32)
    weight = rng.normal(size=(32, 2)).astype(np.float32)
    bias = rng.normal(size=32).astype(np.float32)
    codec = {'residual.0.weight': weight, 'residual.0.bias': bias}
    artifact = {'hidden_weight': weight, 'hidden_bias': bias}
    assert np.array_equal(core.decoder_features(z, last, codec), core.decoder_features(z, last, artifact))
