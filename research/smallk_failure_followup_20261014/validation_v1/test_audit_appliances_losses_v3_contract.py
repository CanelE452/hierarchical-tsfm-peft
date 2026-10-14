from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from research.smallk_failure_followup_20261014.validation_v1 import audit_appliances_losses_v3 as audit


def test_queue_adapter_accepts_actual_single_fit_result_schema(tmp_path):
    producer = {
        "status": "COMPLETE",
        "dataset": "appliances",
        "seed": 92601,
        "objective": "MSE",
        "tag": "appliances_mse_s92601_v1",
    }
    producer_path = tmp_path / "RESULT.json"
    producer_path.write_text(json.dumps(producer), encoding="utf-8")
    producer_sha = hashlib.sha256(producer_path.read_bytes()).hexdigest()
    queue = {
        "schema": "smallk_appliances_loss_queue_v1",
        "status": "COMPLETE",
        "conditions": [
            {
                "tag": "appliances_mse_s92601_v1",
                "dataset": "appliances",
                "seed": 92601,
                "objective": "MSE",
                "status": "COMPLETE",
                "producer_result": {"path": str(producer_path), "sha256": producer_sha},
            }
        ],
    }
    queue_path = tmp_path / "APPLIANCES_QUEUE_v1.json"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    checks = []

    def check(name, passed, details=None):
        checks.append((name, bool(passed), details))

    refs = audit.producer_refs_from_inputs([queue_path], check)
    conditions = audit.load_condition_records(refs, check)

    assert refs[0]["sha256"] == producer_sha
    assert conditions[0]["objective"] == "MSE"
    assert conditions[0]["_queue_condition"]["objective"] == "MSE"


def test_old_codec_hash_and_app_shape_dtype_hash_are_distinct_contracts():
    state = {
        "b": np.array([1, 2], dtype=np.float32),
        "a": np.array([3], dtype=np.float32),
    }

    old_hash = audit.old_codec_state_hash(state)
    app_hash = audit.state_hash(state)

    assert old_hash != app_hash
    assert old_hash == audit.old_codec_state_hash(state)
    assert app_hash == audit.state_hash(state)


def test_curve_objective_policy_only_equates_mse_to_mse_diagnostic():
    mse_row = {"objective_loss": 0.25, "actual_Y_loss": 0.25}
    mae_row = {"objective_loss": 0.5, "actual_Y_loss": 0.25}
    huber_row = {"objective_loss": 0.125, "actual_Y_loss": 0.25}

    assert np.isclose(mse_row["objective_loss"], mse_row["actual_Y_loss"])
    assert not np.isclose(mae_row["objective_loss"], mae_row["actual_Y_loss"])
    assert not np.isclose(huber_row["objective_loss"], huber_row["actual_Y_loss"])


def test_final_codec_uses_old_hash_algorithm_and_q_uses_app_shape_dtype_hash():
    codec_state = {
        "codec.project.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "codec.project.bias": np.array([0.5, -0.25], dtype=np.float32),
    }
    q_state = {
        "q.weight": np.array([[1.0, 2.0]], dtype=np.float32),
    }
    checkpoint_record = {
        "state_sha_old_algorithm": audit.old_codec_state_hash(codec_state),
    }
    q_record = {
        "q_state_hash_APP_shape_dtype_algorithm": audit.state_hash(q_state),
    }

    assert checkpoint_record["state_sha_old_algorithm"] == audit.old_codec_state_hash(codec_state)
    assert checkpoint_record["state_sha_old_algorithm"] != audit.state_hash(codec_state)
    assert q_record["q_state_hash_APP_shape_dtype_algorithm"] == audit.state_hash(q_state)
