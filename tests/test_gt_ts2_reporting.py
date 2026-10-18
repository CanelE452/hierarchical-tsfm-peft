import csv
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def reporter():
    try:
        return importlib.import_module("research.gt_ts2_fastcheck_20261018_v1.reporting")
    except ModuleNotFoundError:
        pytest.fail("The independently rescored fastcheck reporter is missing")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


@pytest.fixture
def campaign(tmp_path):
    public, private = tmp_path / "public", tmp_path / "private"
    public.mkdir()
    private.mkdir()
    write_json(public / "START_STATE.json", {"started_unix": 1000, "parent_commit": "a" * 40,
                                             "GPU_wall_limit_seconds": 5400, "wall_limit_seconds": 10800})
    write_json(public / "EXPERIMENT_CONTRACT.json", {"training": {"final_selection": "fixed update"},
                                                    "limits": {"pretraining_overlap": "UNKNOWN"}})
    write_json(public / "PARENT_BINDINGS.json", {"92601": {"GT": {}, "TS2": {}},
                                               "92602": {"GT": {}, "TS2": {}}})
    write_json(public / "UPDATE_DECISION.json", {"updates": 100, "status": "BOUNDED_100_UPDATE_PILOT",
                                               "selection_uses_VAL": False})
    write_json(private / "GPU_LEDGER.json", {"status": "IDLE", "GPU_seconds": 250,
                                             "stages": [{"stage": "phase_b", "ended_unix": 1400,
                                                         "wall_seconds": 250, "status": "COMPLETE"}]})
    target = np.zeros((2, 48, 2), np.float64)
    mask = np.ones_like(target, bool)
    mask[:, 16::2, 1] = False
    shared = dict(target=target, mask=mask, origins=np.array([100, 124], np.int64),
                  mean=np.array([10., 20.]), std=np.array([2., 4.]))

    def add(run_id, phase, mode, seed, length, arm, value, updates=0):
        prediction = np.full_like(target, value)
        prediction[:, :, 1] *= 2
        path = private / "predictions" / (run_id + ".npz")
        path.parent.mkdir(exist_ok=True)
        np.savez(path, prediction=prediction, **shared)
        receipt = dict(run_id=run_id, status="COMPLETE", phase=phase, mode=mode, seed=seed,
                       L=length, arm=arm, updates=updates, MSE=2.5 * value ** 2, MAE=1.5 * value,
                       prediction_reference="cache://predictions/" + path.name,
                       artifact_sha256=digest(path), data_contract_sha256="b" * 64,
                       origin_count=2, H=48, C=2)
        for key, value_array in {"prediction": prediction, **shared}.items():
            receipt[key + "_sha256"] = array_digest(value_array)
        write_json(public / "receipts" / (run_id + ".json"), receipt)

    for length, value in ((512, 2.), (1024, 1.8), (2048, 1.6)):
        add(f"A_RAW_L{length}", "A", "raw", 0, length, "RAW_F", value)
    arms = ("GT_FT_L512", "GT_FT_L1024", "GT_TS_PRE_L1024", "GT_TS_RAND_L1024")
    for seed in (92601, 92602):
        offset = (seed - 92601) / 10
        for key, mode in (("GT", "gt"), ("TS2", "ts2")):
            add(f"A0_{key}_L512_s{seed}", "A0", mode, seed, 512,
                "HISTORICAL_CALIBRATED_F_REPRODUCTION", 2.5 + offset)
        for length, value in ((512, 2.5), (1024, 2.3), (2048, 2.1)):
            add(f"A_GT_L{length}_s{seed}", "A", "gt", seed, length, "GT_FROZEN", value + offset)
        add(f"A_TS2_L1024_s{seed}", "A", "ts2", seed, 1024, "TS2_NATIVE_D", 2.4 + offset)
        add(f"A_ENSEMBLE_L1024_s{seed}", "A", "ensemble", seed, 1024,
            "OUTPUT_ENSEMBLE_CONTEXT_ONLY", 2.35 + offset)
        for arm, value in zip(arms, (2., 1.8, 1.6, 1.7)):
            length = 512 if arm == "GT_FT_L512" else 1024
            mode = "pre" if "PRE" in arm else "rand" if "RAND" in arm else "gt"
            run_id = f"B_{arm}_s{seed}"
            add(run_id, "B", mode, seed, length, arm, value + offset, 100)
            add(f"CP0_B_{arm}_s{seed}", "CP0", mode, seed, length, arm, 2.3 + offset)
            write_json(public / "training" / (run_id + ".json"), {
                "run_id": run_id, "seed": seed, "updates": 100, "selection_uses_VAL": False,
                "schedule_sha256": str(seed).zfill(64),
                "seen_origin_count": 788, "unique_seen_origins": 197, "observed_target_exposure": 700000,
                "F_unchanged": True, "graph_unchanged": True,
                "curves": [{"update": 1, "train_loss": 1., "finite": True},
                           {"update": 100, "train_loss": .8, "finite": True}],
                "diagnostics": [{"update": 0, "forecast_mse": 2.},
                                {"update": 100, "forecast_mse": 1.5}],
            })
    for batch in (1, 8):
        write_json(public / "cost_receipts" / f"A_RAW_L512_B{batch}.json", {
            "run_id": "A_RAW_L512", "batch": batch,
            "summary": {"median_ms": 10. * batch, "peak_allocated_bytes": 1024,
                        "peak_reserved_bytes": 2048, "full_online_TS2": True,
                        "resident_parameters": 100, "CUDA_resident_parameters": 100,
                        "CPU_resident_parameters": 50, "system_resident_parameters": 150},
            "trials": [{"trial": 0, "whole_ms": 9. * batch}, {"trial": 1, "whole_ms": 11. * batch}],
        })
    return public, private


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_generate_recomputes_fp64_rows_and_keeps_frozen_adapted_cp0_distinct(campaign, tmp_path):
    public, private = campaign
    output = tmp_path / "derived"
    result = reporter().generate(output, source_report=public, source_cache=private)
    runs = read_csv(output / "RUNS.csv")
    assert len(runs) == result["verified_prediction_receipts"] == 33
    raw = next(row for row in runs if row["run_id"] == "A_RAW_L512")
    assert float(raw["MSE"]) == 10.
    assert int(raw["resident_parameters"]) == 100
    assert int(raw["CPU_resident_parameters"]) == 50
    assert json.loads(raw["channel_count"]) == [96, 64]
    assert len(read_csv(output / "CHANNEL_HORIZON_STATS.csv")) == 33 * 2 * 4
    comparisons = read_csv(output / "COMPARISONS.csv")
    pre_rand = next(row for row in comparisons if row["comparison"] == "PRE_VS_RAND" and row["seed"] == "92601")
    assert float(pre_rand["mse_relative_percent"]) == pytest.approx((1.6 ** 2 - 1.7 ** 2) / 1.7 ** 2 * 100)
    assert sum(row["comparison"] == "FINAL_VS_CP0" for row in comparisons) == 8
    assert len(comparisons) == result["comparisons"] == 38
    adapted_raw = [row for row in comparisons if row["comparison"] == "ADAPTED_VS_RAW"]
    assert {(row["candidate"], row["reference"]) for row in adapted_raw} == {
        (f"B_{arm}_s{seed}", "A_RAW_L512" if arm == "GT_FT_L512" else "A_RAW_L1024")
        for seed in (92601, 92602)
        for arm in ("GT_FT_L512", "GT_FT_L1024", "GT_TS_PRE_L1024", "GT_TS_RAND_L1024")
    }
    pre_raw = next(row for row in adapted_raw if row["candidate"] == "B_GT_TS_PRE_L1024_s92601")
    assert float(pre_raw["mse_relative_percent"]) == pytest.approx((1.6 ** 2 - 1.8 ** 2) / 1.8 ** 2 * 100)
    report = (output / "README_KO.md").read_text(encoding="utf-8")
    assert "GT_TS_PRE_L1024: 같은 L1024 Raw보다 MSE가 낮은 seed 2/2개, MAE가 낮은 seed 2/2개" in report
    assert "GT_FT_L512: 같은 L512 Raw보다 MSE가 낮은 seed 0/2개, MAE가 낮은 seed 0/2개" in report
    assert len(read_csv(output / "TRAINING_CURVES.csv")) == 32
    resources = json.loads((output / "RESOURCE_LEDGER.json").read_text(encoding="utf-8"))
    assert resources["GPU_seconds_committed"] == 250
    assert resources["wall_seconds_to_last_recorded_end"] == 400
    assert resources["new_adaptation_fits"] == 8
    assert resources["new_pretraining_fits"] == 0


@pytest.mark.parametrize("updates", [100, 300])
def test_readme_describes_fixed_train_diagnostics_without_claiming_convergence(campaign, tmp_path, updates):
    public, private = campaign
    if updates == 300:
        (public / "OBSERVATION_CONTEXT_STATS.csv").write_text("comparison,origin_count\nRAW_LOOKBACK,1\n", encoding="utf-8")
    write_json(public / "UPDATE_DECISION.json", {"updates": updates, "status": f"FIXED_{updates}_UPDATES",
                                               "selection_uses_VAL": False})
    for path in (public / "training").glob("*.json"):
        value = json.loads(path.read_text())
        value["updates"] = updates
        value["diagnostics"] = [
            {"update": 0, "forecast_mse": 2., "forecast_mae": 1.2,
             "reconstruction_mse": 3., "reconstruction_mae": 1.5},
            {"update": 100, "forecast_mse": 1.5, "forecast_mae": 1.,
             "reconstruction_mse": 3.1, "reconstruction_mae": 1.6},
        ]
        if updates == 300:
            value["diagnostics"].append({"update": 300, "forecast_mse": 1.2, "forecast_mae": .9,
                                         "reconstruction_mse": 3.3, "reconstruction_mae": 1.7})
        write_json(path, value)
        receipt_path = public / "receipts" / path.name
        receipt = json.loads(receipt_path.read_text())
        receipt["updates"] = updates
        write_json(receipt_path, receipt)
    output = tmp_path / f"diagnostics_{updates}"
    result = reporter().generate(output, source_report=public, source_cache=private)
    report = (output / "README_KO.md").read_text(encoding="utf-8")
    assert "300 고정 updates 완료는 수렴을 증명하지 않는다" in report
    assert "checkpoint 선택이나 조기 종료에 사용하지 않은 기술 통계" in report
    assert ("[OBSERVATION_CONTEXT_STATS.csv](OBSERVATION_CONTEXT_STATS.csv)" in report) is (updates == 300)
    if updates == 300:
        assert "비교 양쪽의 물리 과거 모든 cell이 관측된 원점" in report
        assert "Phase A 이후에 기준을 등록한 사후 기술 진단" in report
        assert "전체 371개 VAL 원점의 주 결과를 교체하지 않는다" in report
    assert "run_id" in report and "CP0" in report and "U100" in report and "U300" in report
    diagnostic_rows = [line.split() for line in report.splitlines()
                       if line.startswith("B_") and any(metric in line for metric in
                                                       ("forecast_mse", "forecast_mae", "reconstruction_mse", "reconstruction_mae"))]
    assert len(diagnostic_rows) == 32
    values = {tuple(row[:2]): row[2:] for row in diagnostic_rows}
    forecast = values[("B_GT_TS_PRE_L1024_s92601", "forecast_mse")]
    reconstruction = values[("B_GT_TS_PRE_L1024_s92601", "reconstruction_mse")]
    assert [float(value) for value in forecast[:2]] == [2., 1.5]
    if updates == 300:
        assert [float(value) for value in forecast[2:]] == pytest.approx([1.2, -.3])
        assert [float(value) for value in reconstruction[2:]] == pytest.approx([3.3, .2])
    else:
        assert forecast[2:] == ["NOT_RECORDED", "NOT_AVAILABLE"]
    assert len(result["files"]) == 21


@pytest.mark.parametrize("mutation", ["absolute", "traversal", "artifact_sha", "metric", "array_hash"])
def test_generate_rejects_private_path_leaks_and_unverified_numbers_before_writing(campaign, tmp_path, mutation):
    public, private = campaign
    path = public / "receipts" / "A_RAW_L512.json"
    value = json.loads(path.read_text())
    if mutation == "absolute":
        value["source_file"] = "C:/<PRIVATE>/forecast.npz"
    elif mutation == "traversal":
        value["prediction_reference"] = "cache://predictions/../outside.npz"
    elif mutation == "artifact_sha":
        value["artifact_sha256"] = "f" * 64
    elif mutation == "metric":
        value["MSE"] = 123.
    else:
        value["target_sha256"] = "f" * 64
    write_json(path, value)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        reporter().generate(output, source_report=public, source_cache=private)
    assert not output.exists()


def test_three_figures_use_english_labels_and_source_csvs(campaign, tmp_path):
    public, private = campaign
    output = tmp_path / "figures"
    reporter().generate(output, source_report=public, source_cache=private)
    assert len(list(output.glob("FIG*.png"))) == len(list(output.glob("FIG*.svg"))) == 3
    assert len(list(output.glob("FIG*_source.csv"))) == 3
    lookback = (output / "FIG1_Lookback.svg").read_text(encoding="utf-8")
    adaptation = (output / "FIG2_Adaptation.svg").read_text(encoding="utf-8")
    cost = (output / "FIG3_Cost.svg").read_text(encoding="utf-8")
    assert "GT seed 92601" in lookback and "GT seed 92602" in lookback
    assert "PRE vs RAND" in adaptation
    assert "Whole-route latency (ms)" in cost
    report = (output / "README_KO.md").read_text(encoding="utf-8")
    for topic in ("고정", "추가학습", "RAND", "Raw", "UNKNOWN", "보호 TEST", "신뢰구간"):
        assert topic in report
    assert "2C(L/4)=LC/2로 2배 압축" in report
    assert "고정 원점 그룹 3개를 순환한 20회 반복" in report
    assert "서비스 tail 지연 보장으로 해석하지 않는다" in report


def test_partial_report_keeps_missing_runs_and_resume_queue_explicit(campaign, tmp_path):
    public, private = campaign
    (public / "receipts" / "B_GT_TS_PRE_L1024_s92602.json").unlink()
    output = tmp_path / "partial"
    result = reporter().generate(output, source_report=public, source_cache=private)
    assert result["status"] == "PARTIAL"
    missing = next(row for row in read_csv(output / "JOB_STATUS.csv")
                   if row["run_id"] == "B_GT_TS_PRE_L1024_s92602")
    assert missing["status"].startswith("NOT_RUN")
    assert "B_GT_TS_PRE_L1024_s92602" in (output / "RESUME_QUEUE.md").read_text(encoding="utf-8")


def test_training_receipt_missing_from_an_existing_final_prediction_stays_in_resume_queue(campaign, tmp_path):
    public, private = campaign
    (public / "training" / "B_GT_TS_PRE_L1024_s92602.json").unlink()
    output = tmp_path / "missing_training"
    reporter().generate(output, source_report=public, source_cache=private)
    jobs = read_csv(output / "JOB_STATUS.csv")
    assert any(row["run_id"] == "B_GT_TS_PRE_L1024_s92602" and row["category"] == "TRAINING"
               and row["status"] == "NOT_RUN_TRAINING_RECEIPT" for row in jobs)


@pytest.mark.parametrize("mutation", ["run_spec", "cost_reference", "cost_invalid", "schedule", "resident_count"])
def test_report_rejects_misbound_execution_cost_or_matched_schedule(campaign, tmp_path, mutation):
    public, private = campaign
    if mutation == "run_spec":
        path = public / "receipts" / "A_RAW_L512.json"
        value = json.loads(path.read_text())
        value["L"] = 1024
    elif mutation == "schedule":
        path = public / "training" / "B_GT_TS_PRE_L1024_s92601.json"
        value = json.loads(path.read_text())
        value["schedule_sha256"] = "f" * 64
    else:
        path = public / "cost_receipts" / "A_RAW_L512_B1.json"
        value = json.loads(path.read_text())
        if mutation == "cost_reference":
            value["reference_prediction_sha256"] = "f" * 64
        elif mutation == "resident_count":
            value["summary"]["resident_parameters"] = 101
        else:
            value["validation"] = {"status": "FAIL"}
    write_json(path, value)
    with pytest.raises(ValueError):
        reporter().generate(tmp_path / "invalid", source_report=public, source_cache=private)


def test_public_derived_outputs_replay_byte_identically_in_a_separate_process(campaign, tmp_path):
    import matplotlib

    public, private = campaign
    first, second = tmp_path / "first", tmp_path / "second"
    pollution = {"legend.fontsize": 8, "axes.titlesize": 11}
    with matplotlib.rc_context(pollution):
        result = reporter().generate(first, source_report=public, source_cache=private)
        assert {key: matplotlib.rcParams[key] for key in pollution} == pollution
    command = [sys.executable, "-m", "research.gt_ts2_fastcheck_20261018_v1.reporting", "report",
               "--source-report", str(public), "--source-cache", str(private), "--output-dir", str(second)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert result["files"] == {path.name: digest(path) for path in second.iterdir() if path.is_file()}
    assert {path.name: digest(path) for path in first.iterdir()} == result["files"]
