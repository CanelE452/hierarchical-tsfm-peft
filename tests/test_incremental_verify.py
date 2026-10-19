import csv
import gzip
import hashlib
import importlib
import importlib.util
import json

import numpy as np
import pytest


MODULE = "research.incremental_latent_adaptation_20261019_v1.verify"


def _verify():
    assert importlib.util.find_spec(MODULE) is not None, "independent verifier is missing"
    return importlib.import_module(MODULE)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(value.tobytes()).hexdigest())


def _entry(tmp_path, run_id, method, seed, offset, *, target_offset=0.):
    y = (np.arange(12).reshape(3, 2, 2) / 10 + target_offset).astype(np.float32)
    p = (y + offset).astype(np.float32)
    mask = np.ones_like(y, dtype=bool)
    mask[0, 1, 1] = False
    mask[1, :, 0] = False
    arrays = dict(pred_normalized=p, target_normalized=y, observed=mask,
                  origins=np.array([100, 102, 104], np.int64), channel_ids=np.array(["c0", "c1"]))
    metadata = dict(split="V_select", scaler_sha256="scaler", parent_G_sha256="g-parent",
                    data_sha256="same-data", T_access=False)
    path = tmp_path / (run_id + ".npz")
    np.savez_compressed(path, **arrays, metadata_json=np.asarray(json.dumps(metadata)))
    difference = p.astype(np.float64) - y.astype(np.float64)
    score = dict(mse=float(np.mean([np.square(difference[:, :, c][mask[:, :, c]]).mean() for c in range(2)])),
                 mae=float(np.mean([np.abs(difference[:, :, c][mask[:, :, c]]).mean() for c in range(2)])),
                 n_observed=int(mask.sum()), n_origins=3)
    return dict(run_id=run_id, dataset="jena", method=method, seed=seed,
                prediction_path=str(path), prediction_sha256=_sha(path),
                arrays={key: _identity(value) for key, value in arrays.items()}, score=score, metadata=metadata)


def _manifest(tmp_path, entries):
    report = tmp_path / "report"
    report.mkdir(exist_ok=True)
    (report / "C_DECISIONS.json").write_text(json.dumps(dict(status="SEALED_BEFORE_V",
        decisions={"jena": {"status": "NOT_ADMITTED_NO_INCREMENTAL_EVIDENCE"},
                   "pems32": {"status": "NOT_ADMITTED_NO_INCREMENTAL_EVIDENCE"}})))
    path = tmp_path / "FINAL_MANIFEST.json"
    path.write_text(json.dumps(entries))
    return path, report


def test_independent_origin_statistics_recombine_masked_channel_macro(tmp_path):
    module = _verify()
    g = _entry(tmp_path, "g", "G", None, .5)
    f = _entry(tmp_path, "f", "RESIDUAL_F", 92601, .25)
    manifest, report = _manifest(tmp_path, [g, f])
    result = module.verify(manifest, outdir=report)
    assert result["prediction_validation"]["status"] == "PASS"
    assert result["prediction_validation"]["run_count"] == 2
    assert result["status"] == "PARTIAL", "missing mandatory artifacts must prevent overall PASS"
    with gzip.open(report / "ERROR_SUFFICIENT_STATS.csv.gz", "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 24
    assert not any("target" in key or "prediction" in key for key in rows[0])
    selected = [row for row in rows if row["run_id"] == "f"]
    mse = np.mean([sum(float(r["sse"]) for r in selected if r["channel_index"] == str(c)) /
                   sum(int(r["count"]) for r in selected if r["channel_index"] == str(c)) for c in range(2)])
    assert mse == pytest.approx(f["score"]["mse"], abs=1e-15)
    with (report / "PAIRWISE_EFFECTS.csv").open(newline="", encoding="utf8") as stream:
        pairs = list(csv.DictReader(stream))
    pair = next(row for row in pairs if row["reference_method"] == "G" and row["candidate_method"] == "RESIDUAL_F")
    assert float(pair["delta_MSE"]) == pytest.approx(f["score"]["mse"] - g["score"]["mse"])


def test_verifier_rejects_numeric_access_without_sealed_c_decisions(tmp_path):
    module = _verify()
    path = tmp_path / "FINAL_MANIFEST.json"
    path.write_text(json.dumps([dict(prediction_path="missing.npz")]))
    with pytest.raises(ValueError, match="sealed"):
        module.verify(path, outdir=tmp_path)


def test_changed_prediction_bytes_and_paired_targets_fail(tmp_path):
    module = _verify()
    g = _entry(tmp_path, "g", "G", None, .5)
    f = _entry(tmp_path, "f", "RESIDUAL_F", 92601, .25, target_offset=.1)
    manifest, report = _manifest(tmp_path, [g, f])
    result = module.verify(manifest, outdir=report)
    assert result["status"] == "FAIL"
    assert any("paired" in error.lower() and "target" in error.lower() for error in result["errors"])
    f["prediction_sha256"] = "wrong"
    manifest.write_text(json.dumps([g, f]))
    result = module.verify(manifest, outdir=report)
    assert result["status"] == "FAIL"
    assert any("sha" in error.lower() for error in result["errors"])


def test_cost_summary_is_recomputed_from_all_twenty_raw_trials(tmp_path):
    module = _verify()
    fields = ["cost_id", "phase", "trial", "whole_seconds", "batch_size"]
    with (tmp_path / "COST_TRIALS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for phase, n in (("warmup", 10), ("timed", 20)):
            for trial in range(n):
                writer.writerow(dict(cost_id="one", phase=phase, trial=trial, whole_seconds=(trial + 1) / 1000, batch_size=8))
    cost = tmp_path / "COST.csv"
    cost.write_text("cost_id,batch_size,median_batch_ms,IQR_batch_ms,median_origin_ms,IQR_origin_ms\none,8,10.5,9.5,1.3125,1.1875\n")
    assert module.verify_cost_tables(tmp_path)["status"] == "PASS"
    cost.write_text(cost.read_text().replace("10.5", "10.6"))
    assert module.verify_cost_tables(tmp_path)["status"] == "FAIL"


def test_two_seed_effect_means_keep_seed_specific_auxiliary_fields_na(tmp_path):
    module = _verify()
    entries = [_entry(tmp_path, "g", "G", None, .5),
               _entry(tmp_path, "f1", "RESIDUAL_F", 92601, .2),
               _entry(tmp_path, "f2", "RESIDUAL_F", 92602, .3)]
    entries[1].update(alpha=.2, selected_epoch=1)
    entries[2].update(alpha=.8, selected_epoch=3)
    manifest, report = _manifest(tmp_path, entries)
    result = module.verify(manifest, outdir=report)
    assert result["sufficient_statistics_replay"]["status"] == "PASS"
    with (report / "PAIRWISE_EFFECTS.csv").open(newline="", encoding="utf8") as stream:
        pairs = list(csv.DictReader(stream))
    mean = next(row for row in pairs if row["aggregation"] == "two_seed_mean")
    assert mean["candidate_alpha"] == "NA; see by-seed"
    assert mean["candidate_selected_epoch"] == "NA; see by-seed"
    expected = (entries[1]["score"]["mse"] + entries[2]["score"]["mse"]) / 2
    assert float(mean["candidate_MSE"]) == pytest.approx(expected)


def test_numerical_receipt_is_stable_when_report_coverage_changes(tmp_path):
    module = _verify()
    manifest, report = _manifest(tmp_path, [_entry(tmp_path, "g", "G", None, .5)])
    first = module.verify(manifest, outdir=report)
    numeric_sha = _sha(report / "NUMERICAL_VALIDATION.json")
    stat_sha = _sha(report / "ERROR_SUFFICIENT_STATS.csv.gz")
    (report / "README_KO.md").write_text("a newly completed public report")
    second = module.verify(manifest, outdir=report)
    assert first["status"] == second["status"] == "PARTIAL"
    assert numeric_sha == _sha(report / "NUMERICAL_VALIDATION.json")
    assert stat_sha == _sha(report / "ERROR_SUFFICIENT_STATS.csv.gz")
    assert first["requirement_audit"]["missing"] != second["requirement_audit"]["missing"]
    assert (report / "VALIDATION_REPORT_ATTEMPT_001.json").is_file()
