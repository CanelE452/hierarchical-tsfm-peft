import csv
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from test_gt_ts2_reporting import campaign, array_digest, digest, write_json


def replay_module():
    try:
        return importlib.import_module("research.gt_ts2_fastcheck_20261018_v1.public_replay")
    except ModuleNotFoundError:
        pytest.fail("The public-only statistics and image replay implementation is missing")


@pytest.fixture
def public_report(campaign, tmp_path):
    source, private = campaign
    for path in sorted((source / "receipts").glob("*.json")):
        receipt = json.loads(path.read_text())
        archive = private / "predictions" / (receipt["run_id"] + ".npz")
        with np.load(archive, allow_pickle=False) as loaded:
            values = {key: loaded[key] for key in loaded.files}
        for key in ("prediction", "target", "mask"):
            values[key] = np.tile(values[key], (6, 1, 1))
        values["origins"] = np.arange(100, 388, 24, dtype=np.int64)
        np.savez(archive, **values)
        receipt["artifact_sha256"] = digest(archive)
        receipt["origin_count"] = 12
        for key, value in values.items():
            receipt[key + "_sha256"] = array_digest(value)
        write_json(path, receipt)
        if receipt["phase"] not in ("A", "B"):
            continue
        for batch in (1, 8):
            base = float(batch * 10)
            starts = [0, (12 - batch) // 2, 12 - batch]
            trials = []
            for stage, count in (("warmup", 10), ("timed", 20)):
                for iteration in range(count):
                    start = starts[iteration % 3]
                    trials.append({
                        "stage": stage, "iteration": iteration, "group_index": iteration % 3,
                        "groupstart": start, "batch": batch,
                        "origins_sha256": array_digest(values["origins"][start:start + batch]),
                        "whole_batch_ms": base + iteration + 1,
                        "peak_allocated_MiB": 10. + iteration, "peak_reserved_MiB": 20. + iteration,
                        "RSS_MiB": 30. + iteration, "TS2cache_used": False,
                        "requested_context_length": [receipt["L"]], "actual_context_length": [receipt["L"]],
                    })
            summary = {"p50_ms": base + 10.5, "q25_ms": base + 5.75, "q75_ms": base + 15.25,
                       "iqr_ms": 9.5, "p95_ms": base + 19.05, "repetitions": 20,
                       "peak_allocated_MiB": 29., "peak_reserved_MiB": 39., "RSS_MiB": 49.,
                       "origins_per_second_at_median": 1000 * batch / (base + 10.5),
                       "resident_parameters": 100, "CUDA_resident_parameters": 100,
                       "CPU_resident_parameters": 50, "system_resident_parameters": 150}
            write_json(source / "cost_receipts" / f"{receipt['run_id']}_B{batch}.json", {
                "status": "COMPLETE", "spec": {key: receipt[key] for key in ("run_id", "phase", "mode", "seed", "L", "arm")},
                "batch": batch, "summary": summary, "trials": trials,
                "validation": {"status": "PASS"},
                "reference_prediction_sha256": receipt["prediction_sha256"],
                "reference_artifact_sha256": receipt["artifact_sha256"],
            })
    from research.gt_ts2_fastcheck_20261018_v1 import reporting
    output = tmp_path / "public_derived"
    reporting.generate(output, source_report=source, source_cache=private)
    hidden = tmp_path / "hidden_private_arrays"
    assert private.resolve().is_relative_to(tmp_path.resolve())
    assert hidden.resolve().is_relative_to(tmp_path.resolve())
    private.rename(hidden)
    return output


def mutate_csv(path, change):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    change(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_typed_reader_recovers_json_numbers_booleans_and_structured_cells(tmp_path):
    path = tmp_path / "typed.csv"
    path.write_text('seed,metric,flag,other,values,metadata,empty,name\n92601,0.125,True,false,"[1, 2]","{""a"": 3}",,RAW\n', encoding="utf-8")
    row = replay_module().read_typed_csv(path)[0]
    assert row == dict(seed=92601, metric=.125, flag=True, other=False, values=[1, 2],
                       metadata={"a": 3}, empty=None, name="RAW")


def test_public_statistics_and_six_images_replay_without_loading_private_arrays(public_report, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Public replay must not read private NPZ arrays")
    monkeypatch.setattr(np, "load", forbidden)
    result = replay_module().replay(public_report, tmp_path / "replay")
    assert result["status"] == "PASS"
    assert result["metrics"]["runs_verified"] == 33
    assert result["costs"]["run_batch_pairs_verified"] == 42
    assert result["comparisons"]["pairs_verified"] == 38
    assert len(result["images"]) == 6 and all(row["matches"] for row in result["images"].values())
    assert result["private_arrays_read"] is False
    assert len(list((tmp_path / "replay").iterdir())) == 7


@pytest.mark.parametrize("mutation", ["band_sse", "channel_count", "cost_quantile", "timed_count", "group_cycle"])
def test_public_replay_detects_inconsistent_sufficient_statistics_or_cost_trials(public_report, tmp_path, mutation):
    if mutation in ("band_sse", "channel_count"):
        path = public_report / "CHANNEL_HORIZON_STATS.csv"
        def change(rows):
            key = "sse" if mutation == "band_sse" else "count"
            row = next(row for row in rows if row["horizon_band"] == "1-16")
            row[key] = str(float(row[key]) + 1)
    elif mutation == "cost_quantile":
        path = public_report / "COST.csv"
        def change(rows):
            rows[0]["p50_ms"] = "999"
    else:
        path = public_report / "COST_TRIALS.csv"
        def change(rows):
            first = next(i for i, row in enumerate(rows) if row["stage"] == "timed")
            if mutation == "timed_count":
                rows.pop(first)
            else:
                rows[first]["group_index"] = "2"
    mutate_csv(path, change)
    result = replay_module().replay(public_report, tmp_path / "invalid_replay")
    assert result["status"] == "FAIL" and result["issues"]
    assert json.loads((tmp_path / "invalid_replay" / "REPLAY_CHECK.json").read_text())["status"] == "FAIL"


def test_image_mismatch_keeps_actual_regeneration_and_never_copies_the_reference(public_report, tmp_path):
    reference = public_report / "FIG1_Lookback.png"
    reference.write_bytes(b"corrupt public reference image")
    output = tmp_path / "mismatched_images"
    result = replay_module().replay(public_report, output)
    assert result["status"] == "FAIL"
    assert result["images"][reference.name]["matches"] is False
    assert (output / reference.name).read_bytes().startswith(b"\x89PNG")
    assert (output / reference.name).read_bytes() != reference.read_bytes()


def test_output_guard_rejects_public_source_or_an_existing_attempt(public_report, tmp_path):
    with pytest.raises(ValueError):
        replay_module().replay(public_report, public_report)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        replay_module().replay(public_report, existing)


def test_fresh_process_replay_passes_with_npz_and_campaign_cache_open_forbidden(public_report, tmp_path):
    output = tmp_path / "fresh_process"
    script = """
import os, sys
def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0]).replace('\\\\', '/').lower()
        if path.endswith('.npz') or '/.cache/gt_ts2_fastcheck_20261018_v1/' in path:
            raise AssertionError('Private forecast/cache read forbidden in public replay')
sys.addaudithook(guard)
from research.gt_ts2_fastcheck_20261018_v1.public_replay import main
raise SystemExit(main())
"""
    process = subprocess.run([sys.executable, "-c", script, "--source-report", str(public_report),
                              "--output-dir", str(output)], capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    result = json.loads((output / "REPLAY_CHECK.json").read_text())
    assert result["status"] == "PASS"
    assert all(digest(output / name) == digest(public_report / name) for name in result["images"])
