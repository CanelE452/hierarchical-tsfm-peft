from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


VALIDATION = Path(__file__).resolve().parent
sys.path.insert(0, str(VALIDATION))
import audit_stage_native_v1 as audit  # noqa: E402


def save_npz_record(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {"path": str(path), "sha256": audit.sha_file(path)}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"path": str(path), "sha256": audit.sha_file(path)}


def make_split(n: int = 1, k: int = 2, c: int = 1):
    latent = np.zeros((n, 512, k), dtype=np.float32)
    target = np.zeros((n, 48, c), dtype=np.float32)
    observed = np.ones_like(target, dtype=np.bool_)
    raw_last = np.zeros((n, c), dtype=np.float32)
    return {
        "latent_past": latent,
        "z_future": np.zeros((n, 48, k), dtype=np.float32),
        "z_last": np.zeros((n, 1, k), dtype=np.float32),
        "raw_last": raw_last,
        "target": target,
        "observed": observed,
        "origins": np.arange(n, dtype=np.int64),
        "history_observed": np.ones((n, 512, c), dtype=np.bool_),
        "raw_forecast": target.copy(),
        "mean": np.zeros(c, dtype=np.float32),
        "std": np.ones(c, dtype=np.float32),
        "original_prediction": target.copy(),
        "source_gpu_prediction": target.copy(),
    }


def make_feature_artifact(tmp_path: Path, fit_latent: np.ndarray, k: int):
    raw = audit.fixed_past_features(fit_latent)
    center = np.zeros(raw.shape[1], dtype=np.float64)
    scale = np.ones(raw.shape[1], dtype=np.float64)
    fit_raw_sha = audit.array_sha(raw)
    feature_sha = audit.feature_state_sha(center, scale, k, len(raw), fit_raw_sha)
    artifact = save_npz_record(tmp_path / "SHARED_FIT_FEATURE_STATE.npz", center=center, scale=scale)
    return {
        "sha256": feature_sha,
        "latent_channels": k,
        "fit_rows": len(raw),
        "fit_raw_features_sha256": fit_raw_sha,
        "artifact": artifact,
    }


def make_solution_artifact(tmp_path: Path, feature_sha: str, feature_count: int, c: int):
    arrays = {
        "coefficients": np.zeros((feature_count, 48, c), dtype=np.float64),
        "intercept": np.zeros((48, c), dtype=np.float64),
        "observed_counts": np.ones((48, c), dtype=np.int64),
        "channel_counts": np.array([48], dtype=np.int64),
        "normal_equation_denominators": np.ones((48, c), dtype=np.float64),
        "design_ranks": np.ones((48, c), dtype=np.int64),
        "regularized_gram_conditions": np.ones((48, c), dtype=np.float64),
    }
    sol_meta = {
        "kind": "N2_DIRECT_C",
        "alpha": 0.0,
        "with_intercept": True,
        "feature_state_sha256": feature_sha,
    }
    sol_meta["solution_sha256"] = audit.solution_sha(sol_meta, arrays)
    artifact = save_npz_record(tmp_path / "N2_SELECTED_COEFFICIENTS.npz", **arrays)
    model_sha = audit.native_model_sha(feature_sha, sol_meta["solution_sha256"])
    return artifact, sol_meta, model_sha


def make_minimal_n2_producer(tmp_path: Path, *, score_override=None):
    k, c = 2, 1
    splits = {split: make_split(k=k, c=c) for split in ("fit", "tune", "eval")}
    artifacts = []
    for split, arrays in splits.items():
        artifacts.append(save_npz_record(tmp_path / f"INPUT_{split}.npz", **arrays))
    feature = make_feature_artifact(tmp_path, splits["fit"]["latent_past"], k)
    coefficient_artifact, sol_meta, model_sha = make_solution_artifact(tmp_path, feature["sha256"], 33 * k, c)
    zero_score = audit.observed_score(splits["fit"]["target"], splits["fit"]["target"], splits["fit"]["observed"])
    if score_override is not None:
        row_score = dict(zero_score)
        row_score.update(score_override)
    else:
        row_score = zero_score
    alpha_rows = []
    for index, alpha in enumerate(audit.ALPHAS):
        fit_artifact = save_npz_record(tmp_path / f"N2_alpha_{index}_fit.npz", prediction=splits["fit"]["target"], origins=splits["fit"]["origins"])
        tune_artifact = save_npz_record(tmp_path / f"N2_alpha_{index}_tune.npz", prediction=splits["tune"]["target"], origins=splits["tune"]["origins"])
        alpha_rows.append({
            "alpha": alpha,
            "model_sha256": model_sha,
            "solution": sol_meta,
            "coefficient_artifact": coefficient_artifact,
            "fit_scores": {"normalized": row_score},
            "tune_scores": {"normalized": zero_score},
            "fit_artifact": fit_artifact,
            "tune_artifact": tune_artifact,
            "tune_actual_Y_mse": zero_score["mse"],
        })
    selected_artifacts = {}
    split_scores = {}
    for split, arrays in splits.items():
        selected_artifacts[split] = save_npz_record(
            tmp_path / f"N2_SELECTED_{split}.npz",
            prediction=arrays["target"],
            target=arrays["target"],
            observed=arrays["observed"],
            origins=arrays["origins"],
            mean=arrays["mean"],
            std=arrays["std"],
            raw_last=arrays["raw_last"],
        )
        split_scores[split] = {"normalized": zero_score, "physical_per_channel_diagnostic": zero_score}
    decoder_artifact = save_npz_record(
        tmp_path / "SOURCE_DECODER.npz",
        **{
            "codec.decode.weight": np.zeros((c, k), dtype=np.float32),
            "codec.decode.bias": np.zeros(c, dtype=np.float32),
            "residual.0.weight": np.zeros((32, k), dtype=np.float32),
            "residual.0.bias": np.zeros(32, dtype=np.float32),
            "residual.2.weight": np.zeros((c, 32), dtype=np.float32),
            "residual.2.bias": np.zeros(c, dtype=np.float32),
        },
    )
    producer = {
        "status": "COMPLETE",
        "dataset": "beijing_aotizhongxin",
        "seed": 92601,
        "F_loaded": False,
        "F_forward_replayed": False,
        "E_forward_replayed": False,
        "oracle_is_deployable": False,
        "physical_macro_is_primary": False,
        "old_E_D_F_updated": False,
        "codec_before": {"old_algorithm": "old", "bi_algorithm": "bi"},
        "codec_after": {"old_algorithm": "old", "bi_algorithm": "bi"},
        "codec_terminal_hash": {"old_algorithm": "old", "bi_algorithm": "bi"},
        "artifacts": artifacts,
        "source_records": [],
        "source_snapshots": [],
        "feature_state": feature,
        "source_decoder_artifact": decoder_artifact,
        "families": {
            "N2": {
                "status": "COMPLETE",
                "alpha_grid": alpha_rows,
                "selected_index": 0,
                "selected_alpha": 0.0,
                "selected_model_sha256": model_sha,
                "selected_coefficient_artifact": coefficient_artifact,
                "selected_model_metadata": {"kind": "N2_DIRECT_C", "alpha": 0.0, "solution": sol_meta, "fixed_n1_model_sha256": None},
                "selected_artifacts": selected_artifacts,
                "splits": split_scores,
            }
        },
    }
    record = write_json(tmp_path / "RESULT.json", producer)
    return Path(record["path"])


def test_describe_is_source_only():
    info = audit.describe()
    assert info["status"] == "SOURCE_ONLY_NOT_EXECUTED"
    assert info["no_actual_reads_on_import_or_describe"] is True


def test_minimal_n2_producer_reconstructs_scores_selection_and_model_hash(tmp_path):
    path = make_minimal_n2_producer(tmp_path)

    row = audit.audit_one(path)

    assert row["status"] == "PASS", row["failures"]
    assert row["candidate_rows_checked"] == 7
    assert row["selected_predictions_checked"] == 3
    assert row["family_status"]["N2"] == "COMPLETE"


def test_score_mismatch_is_reported_without_claiming_pass(tmp_path):
    path = make_minimal_n2_producer(tmp_path, score_override={"mse": 123.0})

    row = audit.audit_one(path)

    assert row["status"] == "FAIL"
    assert any(item.startswith("SCORE_MSE_MISMATCH:N2:0:fit") for item in row["failures"])


def test_execute_requires_exact_four_producers(tmp_path):
    path = make_minimal_n2_producer(tmp_path / "one")

    try:
        audit.execute([path], out=tmp_path / "AUDIT.json")
    except audit.AuditFailure as error:
        assert "EXACT_FOUR_STAGE_NATIVE_PRODUCERS_REQUIRED" in str(error)
    else:
        raise AssertionError("execute accepted a partial producer set")


def test_queue_conditions_resolve_producer_result_paths(tmp_path):
    path = make_minimal_n2_producer(tmp_path / "producer")
    queue = tmp_path / "QUEUE.json"
    queue.write_text(json.dumps({"conditions": [{"producer_result": {"path": str(path), "sha256": audit.sha_file(path)}}]}), encoding="utf-8")

    paths = audit.producer_paths_from_queue(queue)

    assert paths == [path]
