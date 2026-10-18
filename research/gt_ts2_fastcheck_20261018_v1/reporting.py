"""Independently rescore private forecasts and replay public report artifacts."""
from __future__ import annotations

import argparse
import csv
from io import BytesIO, StringIO
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

import numpy as np

from . import analysis as a
from . import common as c


SEEDS = (92601, 92602)
ARMS = ("GT_FT_L512", "GT_FT_L1024", "GT_TS_PRE_L1024", "GT_TS_RAND_L1024")
FIELDS = ("prediction", "target", "mask", "origins", "mean", "std")


def _public(value):
    if isinstance(value, dict):
        for key, item in value.items():
            _public(str(key))
            _public(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _public(item)
    elif isinstance(value, str):
        if (PureWindowsPath(value).is_absolute() or value.startswith("/")
                or re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", value)):
            raise ValueError("Public receipt contains a local absolute path")
    elif isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Public receipt contains a nonfinite numeric value")


def _read(path, default=None):
    if not path.exists():
        return default
    value = c.read_json(path)
    _public(value)
    return value


def _expected():
    specs = []
    for length in (512, 1024, 2048):
        specs.append(dict(run_id=f"A_RAW_L{length}", phase="A", mode="raw", seed=0, L=length, arm="RAW_F"))
    for seed in SEEDS:
        for key, mode in (("GT", "gt"), ("TS2", "ts2")):
            specs.append(dict(run_id=f"A0_{key}_L512_s{seed}", phase="A0", mode=mode, seed=seed, L=512))
        for length in (512, 1024, 2048):
            specs.append(dict(run_id=f"A_GT_L{length}_s{seed}", phase="A", mode="gt", seed=seed, L=length))
        for mode in ("TS2", "ENSEMBLE"):
            specs.append(dict(run_id=f"A_{mode}_L1024_s{seed}", phase="A", mode=mode.lower(), seed=seed, L=1024))
        for arm in ARMS:
            for prefix, phase in (("B_", "B"), ("CP0_B_", "CP0")):
                specs.append(dict(run_id=f"{prefix}{arm}_s{seed}", phase=phase, seed=seed,
                                  L=512 if arm == "GT_FT_L512" else 1024, arm=arm))
    return specs


def _predictions(source, cache):
    receipts, bundles, runs = {}, {}, []
    expected = {row["run_id"]: row for row in _expected()}
    for path in sorted((source / "receipts").glob("*.json")):
        receipt = _read(path)
        run_id = receipt["run_id"]
        if path.stem != run_id or run_id in receipts:
            raise ValueError("Duplicate or mismatched prediction receipt identity")
        if run_id not in expected or any(receipt.get(key) != value for key, value in expected[run_id].items()):
            raise ValueError("Prediction receipt differs from the sealed run identity")
        receipts[run_id] = receipt
        if receipt.get("status") != "COMPLETE":
            continue
        reference = receipt["prediction_reference"]
        prefix = "cache://predictions/"
        if not reference.startswith(prefix):
            raise ValueError("Prediction reference must use the private campaign cache scheme")
        name = reference[len(prefix):]
        if PurePosixPath(name).name != name or "\\" in name or not name.endswith(".npz"):
            raise ValueError("Prediction reference escapes the allowlisted predictions directory")
        archive = (cache / "predictions" / name).resolve()
        if not archive.is_relative_to((cache / "predictions").resolve()):
            raise ValueError("Private prediction symlink escapes this campaign")
        if c.file_sha(archive) != receipt["artifact_sha256"]:
            raise ValueError("Private prediction archive SHA256 differs")
        with np.load(archive, allow_pickle=False) as values:
            if any(key not in values for key in FIELDS):
                raise ValueError("Private forecast lacks a cohort or scaling field")
            bundle = {key: values[key] for key in FIELDS}
        for key, value in bundle.items():
            if c.array_sha(value) != receipt[key + "_sha256"]:
                raise ValueError("Private prediction array fingerprint differs: " + key)
        prediction = bundle["prediction"]
        if prediction.shape != (receipt["origin_count"], receipt["H"], receipt["C"]):
            raise ValueError("Receipt forecast shape differs")
        metrics = a.score(prediction, bundle["target"], bundle["mask"])
        for upper, lower in (("MSE", "mse"), ("MAE", "mae")):
            if not np.isfinite(receipt[upper]) or not np.isclose(receipt[upper], metrics[lower], rtol=1e-12, atol=1e-12):
                raise ValueError("Independent FP64 metric disagrees with receipt: " + run_id)
        row = {**receipt, "MSE": metrics["mse"], "MAE": metrics["mae"],
               **{key: metrics[key] for key in ("channel_sse", "channel_sae", "channel_count", "channel_mse", "channel_mae")},
               "metric_validation": "FP64_RECOMPUTED_PASS"}
        bundles[run_id] = bundle
        runs.append(row)
    return receipts, bundles, runs


def _comparisons():
    result = []
    for length in (1024, 2048):
        result.append(("RAW_LOOKBACK", 0, f"A_RAW_L{length}", "A_RAW_L512"))
    for seed in SEEDS:
        for length in (1024, 2048):
            result.append(("FROZEN_GT_LOOKBACK", seed, f"A_GT_L{length}_s{seed}", f"A_GT_L512_s{seed}"))
        for length in (512, 1024, 2048):
            result.append(("FROZEN_GT_VS_RAW", seed, f"A_GT_L{length}_s{seed}", f"A_RAW_L{length}"))
        for key in ("GT", "TS2"):
            result.append(("OUTPUT_ENSEMBLE_VS_" + key, seed, f"A_ENSEMBLE_L1024_s{seed}", f"A_{key}_L1024_s{seed}"))
        result.extend([
            ("ADAPTED_GT_LOOKBACK", seed, f"B_GT_FT_L1024_s{seed}", f"B_GT_FT_L512_s{seed}"),
            ("PRE_VS_GT", seed, f"B_GT_TS_PRE_L1024_s{seed}", f"B_GT_FT_L1024_s{seed}"),
            ("PRE_VS_RAND", seed, f"B_GT_TS_PRE_L1024_s{seed}", f"B_GT_TS_RAND_L1024_s{seed}"),
        ])
        for arm in ARMS:
            result.append(("FINAL_VS_CP0", seed, f"B_{arm}_s{seed}", f"CP0_B_{arm}_s{seed}"))
            length = 512 if arm == "GT_FT_L512" else 1024
            result.append(("ADAPTED_VS_RAW", seed, f"B_{arm}_s{seed}", f"A_RAW_L{length}"))
    return result


def _paired_rows(receipts, bundles):
    comparisons, origins = [], []
    for name, seed, candidate, reference in _comparisons():
        if candidate not in bundles or reference not in bundles:
            continue
        if receipts[candidate]["data_contract_sha256"] != receipts[reference]["data_contract_sha256"]:
            raise ValueError("Compared runs have different data contracts")
        summary, rows = a.paired(bundles[candidate], bundles[reference])
        identity = dict(comparison=name, seed=seed, candidate=candidate, reference=reference)
        comparisons.append({**identity, **summary, "status": "PAIRED_CONTRACT_PASS"})
        origins.extend({"row_type": "PAIRED", **identity, **row} for row in rows)
    return comparisons, origins


def _training(source, receipts, decision):
    summaries, rows = [], []
    for path in sorted((source / "training").glob("*.json")):
        value = _read(path)
        run_id = value.get("run_id", path.stem)
        if not run_id.startswith("B_") or "CP0" in run_id:
            continue
        if value.get("selection_uses_VAL") is not False:
            raise ValueError("Forecast adaptation must use the registered final update")
        if decision.get("updates") is not None and value["updates"] != decision["updates"]:
            raise ValueError("Phase B arm used a different update budget")
        if value.get("F_unchanged", value.get("F_state_unchanged")) is not True or value.get("graph_unchanged") is not True:
            raise ValueError("Frozen foundation/spatial graph evidence failed")
        if run_id in receipts and receipts[run_id].get("updates") != value["updates"]:
            raise ValueError("Final prediction and training checkpoint update differ")
        summaries.append(value)
        for key, row_type in (("curves", "MINIBATCH"), ("diagnostics", "FIXED_TRAIN_DIAGNOSTIC")):
            rows.extend({"run_id": run_id, "seed": value["seed"], "row_type": row_type,
                         "fixed_updates": value["updates"], **row} for row in value.get(key, []))
    for seed in SEEDS:
        same_seed = [row for row in summaries if int(row["seed"]) == seed]
        hashes = {row["schedule_sha256"] for row in same_seed if "schedule_sha256" in row}
        if len(hashes) > 1:
            raise ValueError("Matched Phase B arms used different sampling schedules")
    return summaries, rows


def _costs(source, run_index):
    rows, trials = [], []
    for path in sorted((source / "cost_receipts").glob("*.json")):
        value = _read(path)
        if value.get("status", "COMPLETE") != "COMPLETE":
            continue
        spec = value.get("spec", {})
        run_id = value.get("run_id", spec.get("run_id"))
        if run_id not in run_index:
            raise ValueError("Cost receipt has no verified prediction cohort")
        if spec and spec.get("run_id") != run_id:
            raise ValueError("Cost receipt and model specification differ")
        if value.get("validation", {}).get("status", "PASS") != "PASS":
            raise ValueError("Cost measurement validation failed")
        for receipt_key, prediction_key in (("reference_prediction_sha256", "prediction_sha256"),
                                             ("reference_artifact_sha256", "artifact_sha256")):
            if receipt_key in value and value[receipt_key] != run_index[run_id][prediction_key]:
                raise ValueError("Cost measurement is bound to a different prediction artifact")
        batch = int(value["batch"])
        if batch not in (1, 8):
            raise ValueError("Unregistered cost batch size")
        identity = {key: run_index[run_id].get(key) for key in ("run_id", "phase", "mode", "seed", "L", "arm", "MSE", "MAE")}
        identity["batch"] = batch
        rows.append({**identity, **value["summary"],
                     "cost_receipt": "cost_receipts/" + path.name,
                     "loaded_baseline": value.get("loaded_baseline"), "warmup_peaks": value.get("warmup_peaks"),
                     "validation": value.get("validation")})
        trials.extend({**identity, **trial} for trial in value.get("trials", []))
    keys = [(row["run_id"], row["batch"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate run/batch cost receipts")
    return rows, trials


def _join_resident(runs, costs):
    for run in runs:
        measured = [row for row in costs if row["run_id"] == run["run_id"] and "resident_parameters" in row]
        if not measured:
            continue
        inventories = [{key: value for key, value in row.items()
                        if key == "resident_inventory" or "resident_parameter" in key or "resident_buffer" in key}
                       for row in measured]
        if any(inventory != inventories[0] for inventory in inventories[1:]):
            raise ValueError("Run's batch1/batch8 measured resident parameter inventories differ")
        if "resident_parameters" in run:
            run["inference_receipt_resident_parameters"] = run["resident_parameters"]
        run.update(inventories[0])
        run["resident_parameters_source"] = [row["cost_receipt"] for row in measured]
        run["resident_parameters_definition"] = "Measured unique CUDA resident Parameter objects; CPU/system counts are separate; F counted once"


def _csv(rows, first=()):
    fields = list(first) + sorted({key for row in rows for key in row} - set(first))
    if not fields:
        fields = ["status"]
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
                         if isinstance(value, (list, dict, tuple)) else value for key, value in row.items()})
    return stream.getvalue().encode("utf-8")


def _figure_bytes(runs, comparisons, costs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs = {}
    colors = ("#0072B2", "#D55E00")
    style = {"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
             "svg.hashsalt": "gt_ts2_fastcheck_20261018_v1", "axes.spines.top": False,
             "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white"}

    def save(fig, name, source):
        fig.subplots_adjust(bottom=.24, top=.86, wspace=.32)
        for suffix in ("png", "svg"):
            buffer = BytesIO()
            metadata = {"Date": None, "Creator": "gt_ts2_fastcheck_20261018_v1"} if suffix == "svg" else {"Software": "gt_ts2_fastcheck_20261018_v1"}
            fig.savefig(buffer, format=suffix, dpi=160, metadata=metadata)
            outputs[name + "." + suffix] = buffer.getvalue()
        outputs[name + "_source.csv"] = _csv(source)
        plt.close(fig)

    with matplotlib.rc_context({**matplotlib.rcParamsDefault, **style, "backend": "Agg"}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        source = [row for row in runs if row["phase"] == "A" and row["mode"] in ("raw", "gt")]
        for axis, metric in zip(axes, ("MSE", "MAE")):
            for seed, color in zip((0, *SEEDS), ("#222222", *colors)):
                selected = sorted([row for row in source if row["seed"] == seed], key=lambda row: row["L"])
                if selected:
                    axis.plot([row["L"] for row in selected], [row[metric] for row in selected],
                              marker="D" if seed == 0 else "o", color=color,
                              label="Raw" if seed == 0 else f"GT seed {seed}")
            axis.set(xlabel="Physical lookback L", ylabel="Normalized channel-macro " + metric,
                     xticks=[512, 1024, 2048])
            axis.grid(alpha=.2)
        fig.suptitle("Frozen GT and Raw: identical origins and targets")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        save(fig, "FIG1_Lookback", source)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
        source = [row for row in runs if row["phase"] == "B" and int(row.get("updates") or 0) > 0]
        for axis, metric in zip(axes[:2], ("MSE", "MAE")):
            for seed, color in zip(SEEDS, colors):
                points = {row["arm"]: row for row in source if row["seed"] == seed}
                x = [i for i, arm in enumerate(ARMS) if arm in points]
                if x:
                    axis.plot(x, [points[ARMS[i]][metric] for i in x], marker="o", color=color, label=f"Seed {seed}")
            axis.set(xticks=range(4), xticklabels=["GT512", "GT1024", "PRE", "RAND"],
                     ylabel="Normalized channel-macro " + metric)
            axis.grid(alpha=.2)
        names = ("ADAPTED_GT_LOOKBACK", "PRE_VS_GT", "PRE_VS_RAND")
        pair_source = [row for row in comparisons if row["comparison"] in names]
        for seed, color in zip(SEEDS, colors):
            points = {row["comparison"]: row for row in pair_source if row["seed"] == seed}
            x = [i for i, name in enumerate(names) if name in points]
            if x:
                axes[2].plot(x, [points[names[i]]["mse_relative_percent"] for i in x], marker="o", color=color)
        axes[2].axhline(0, color="#777777", linewidth=.8)
        axes[2].set(xticks=range(3), xticklabels=["GT1024 vs GT512", "PRE vs GT1024", "PRE vs RAND"],
                    ylabel="Paired MSE change (%)")
        axes[2].tick_params(axis="x", labelrotation=18)
        axes[2].grid(alpha=.2)
        fig.suptitle("Forecast adaptation: fixed final update; negative paired change is better")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
        save(fig, "FIG2_Adaptation", [dict(row_type="FINAL", **row) for row in source]
             + [dict(row_type="PAIRED", **row) for row in pair_source])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        families = {"raw": ("Raw", "#222222", "D"), "gt_frozen": ("Frozen GT", "#D55E00", "o"),
                    "gt": ("Adapted GT", "#009E73", "s"), "pre": ("PRE", "#0072B2", "*"),
                    "rand": ("RAND", "#CC79A7", "P"), "ts2": ("TS2 native D", "#56B4E9", "^"),
                    "ensemble": ("Output ensemble", "#777777", "X")}
        for axis, batch in zip(axes, (1, 8)):
            for family, (label, color, marker) in families.items():
                selected = [row for row in costs if row["batch"] == batch and
                            ("gt_frozen" if row["phase"] == "A" and row["mode"] == "gt" else row["mode"]) == family]
                if selected:
                    for i, row in enumerate(selected):
                        axis.scatter(row.get("p50_ms", row.get("median_ms")), row["MSE"],
                                     facecolors="none" if row["seed"] == SEEDS[1] else color,
                                     edgecolors=color, marker=marker, s=65,
                                     label=label if i == 0 else None, alpha=.8)
            axis.set(xlabel="Whole-route latency (ms)", ylabel="Normalized channel-macro MSE", title=f"Batch {batch}")
            axis.grid(alpha=.2)
        fig.suptitle("Accuracy and measured whole-route cost; online encoders included")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            from matplotlib.lines import Line2D
            handles += [Line2D([], [], marker="o", linestyle="none", color="#222222"),
                        Line2D([], [], marker="o", linestyle="none", color="#222222", markerfacecolor="none")]
            labels += ["Seed 92601 (filled)", "Seed 92602 (open)"]
            fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
        save(fig, "FIG3_Cost", costs)
    return outputs


def _percent_line(row):
    return (f"seed {row['seed']}: MSE {row['candidate_mse']:.8g} 대 {row['reference_mse']:.8g}, "
            f"변화 {row['mse_relative_percent']:+.4f}%; MAE 변화 {row['mae_relative_percent']:+.4f}%")


def _readme(status, runs, comparisons, costs, training, decision, resources, observation_context_stats=False):
    lines = ["# GTConvAE 긴 관측 과거와 TS2Vec 잠재 결합 빠른 검증", "",
             f"전체 상태: **{status}**. 검증된 예측 {len(runs)}/33개, 최종 추가학습 {len(training)}/8개, 비용 {len(costs)}/42개.", "",
             "## 질문 1. 고정한 GT에 긴 과거를 주면 예측이 좋아지는가", "",
             "[확인] 아래 수치는 같은 VAL 원점·정답·관측 mask·TRAIN 정규화에서 독립 FP64로 다시 계산했다. 음수 변화는 오차 감소다.", ""]
    frozen = [row for row in comparisons if row["comparison"] == "FROZEN_GT_LOOKBACK"]
    lines.extend(f"- {row['candidate']} 대 {row['reference']}: {_percent_line(row)}" for row in frozen)
    if not frozen:
        lines.append("해당 대응 결과가 아직 없어 긴 입력의 효과를 판단하지 않는다.")
    for length in (1024, 2048):
        same_length = [row for row in frozen if f"_L{length}_" in row["candidate"]]
        if len(same_length) == 2:
            lowered = sum(row["mse_delta"] < 0 for row in same_length)
            lines.append(f"L{length}의 고정 GT MSE가 자기 L512보다 낮은 seed는 {lowered}/2개다. 이 결과는 기존 부모에 긴 관측 범위를 준 효과이며 새 reconstruction 사전학습 결과가 아니다.")
    lines += ["", "Raw의 같은 길이 변화와 GT 대 Raw의 같은 길이 비교를 함께 본다. GT가 자기 L512보다 좋아졌다는 사실만으로 실용 기준선을 이겼다고 말하지 않는다.", ""]
    lines.extend(f"- {row['candidate']} 대 {row['reference']}: MSE 변화 {row['mse_relative_percent']:+.4f}%"
                 for row in comparisons if row["comparison"] in ("RAW_LOOKBACK", "FROZEN_GT_VS_RAW"))
    raw_comparisons = [row for row in comparisons if row["comparison"] == "FROZEN_GT_VS_RAW"]
    if len(raw_comparisons) == 6:
        lowered = sum(row["mse_delta"] < 0 for row in raw_comparisons)
        lines.append(f"같은 길이 Raw보다 고정 GT의 MSE가 낮은 길이×seed 조합은 {lowered}/6개다. 자기 이전 결과의 개선과 Raw 기준선 초과를 구분한다.")
    lines += ["", "## 질문 2. 같은 추가학습 예산에서 TS2Vec 사전학습 표현이 도움이 되는가", "",
              f"사전 고정 예산: {decision.get('status', 'NOT_RUN')} / 최종 {decision.get('updates')} updates. 모든 arm은 같은 seed별 전체 TRAIN 원점 순서를 쓰고 마지막 작은 batch를 유지한다. VAL로 최선 checkpoint를 고르지 않는다.", ""]
    adapted = [row for row in comparisons if row["comparison"] in ("ADAPTED_GT_LOOKBACK", "PRE_VS_GT", "PRE_VS_RAND")]
    lines.extend(f"- {row['comparison']}: {_percent_line(row)}" for row in adapted)
    if not adapted:
        lines.append("대응 arm의 최종 결과가 없어 잠재 결합의 효과를 판단하지 않는다.")
    pre_rand = [row for row in adapted if row["comparison"] == "PRE_VS_RAND"]
    pre_gt = [row for row in adapted if row["comparison"] == "PRE_VS_GT"]
    if len(pre_rand) == len(pre_gt) == 2:
        if all(row["mse_delta"] < 0 for row in pre_rand + pre_gt):
            lines.append("두 seed 모두 PRE가 GT1024와 RAND보다 낮은 MSE를 보였다. 이 한정 예산·연결층·데이터 조건에서 사전학습 표현의 추가 검토 근거다.")
        else:
            lines.append("PRE가 GT1024와 RAND를 두 seed에서 모두 이긴다는 근거는 확보되지 않았다. seed별 차이와 비용을 유지해 해석한다.")
    lines += ["", "최종 Phase B 모델과 같은 L의 Raw 기준선을 별도로 비교한다. CP0 대비 변화는 FINAL_VS_CP0에, 아래 Raw 대비 변화는 ADAPTED_VS_RAW에 기록한다.", ""]
    adapted_raw = [row for row in comparisons if row["comparison"] == "ADAPTED_VS_RAW"]
    lines.extend(f"- {row['candidate']} 대 {row['reference']}: {_percent_line(row)}" for row in adapted_raw)
    for arm in ARMS:
        rows = [row for row in adapted_raw if row["candidate"].startswith(f"B_{arm}_s")]
        if len(rows) == 2:
            length = 512 if arm == "GT_FT_L512" else 1024
            mse_lower = sum(row["mse_delta"] < 0 for row in rows)
            mae_lower = sum(row["mae_delta"] < 0 for row in rows)
            mse_equal = sum(row["mse_delta"] == 0 for row in rows)
            mae_equal = sum(row["mae_delta"] == 0 for row in rows)
            lines.append(f"{arm}: 같은 L{length} Raw보다 MSE가 낮은 seed {mse_lower}/2개, MAE가 낮은 seed {mae_lower}/2개다. "
                         f"MSE 동일/높음 {mse_equal}/{2-mse_lower-mse_equal}개, MAE 동일/높음 {mae_equal}/{2-mae_lower-mae_equal}개다.")
    if not adapted_raw:
        lines.append("같은 길이 Raw와 대응 가능한 최종 Phase B 결과가 아직 없다.")
    lines += ["", "고정 GT의 추론과 예측 목적 추가학습은 서로 다른 질문이다. A0는 과거 loader의 전처리 재현이며 새 창별 전처리 비교에서 제외한다. CP0와 최종 모델을 모두 공개하고, 출력 0.5 평균은 보조 대조이며 잠재 결합 성공을 대신하지 않는다.", "",
              "두 seed 산술평균도 함께 남긴다. 아래 평균은 반복 두 개의 요약이며 신뢰구간이 아니다.", "", "```text", "group                         seeds   MSE_mean        MAE_mean"]
    groups = [(f"Frozen_GT_L{length}", [row for row in runs if row["phase"] == "A" and row["mode"] == "gt" and row["L"] == length])
              for length in (512, 1024, 2048)]
    groups += [(arm, [row for row in runs if row["phase"] == "B" and row.get("arm") == arm]) for arm in ARMS]
    for label, rows in groups:
        if len(rows) == 2:
            lines.append(f"{label:<29} 2       {np.mean([row['MSE'] for row in rows]):<15.9g} {np.mean([row['MAE'] for row in rows]):.9g}")
    lines += ["```", "", "출력 평균의 보조 결과:", ""]
    ensemble_rows = [row for row in comparisons if row["comparison"].startswith("OUTPUT_ENSEMBLE")]
    lines.extend(f"- {row['comparison']}: {_percent_line(row)}" for row in ensemble_rows)
    if not ensemble_rows:
        lines.append("해당 보조 대조가 아직 완료되지 않았다.")
    lines += ["", "## 고정 TRAIN 진단과 수렴 해석", "",
              "300 고정 updates 완료는 수렴을 증명하지 않는다. 아래 CP0·100·300점은 같은 고정 TRAIN 진단 원점에서 측정하고 checkpoint 선택이나 조기 종료에 사용하지 않은 기술 통계다. 미니배치 loss와 구분하며, 과거 재구성 점수는 보조 진단이다. 100→300 변화는 U300−U100의 절대 차이이고 음수는 해당 오차 감소다. 기록하지 않은 진단점과 변화는 수치로 채우지 않는다.", "",
              "```text", "run_id                              metric                 CP0             U100            U300            delta_100_to_300"]
    for value in training:
        diagnostics = {int(row["update"]): row for row in value.get("diagnostics", [])}
        for metric in ("forecast_mse", "forecast_mae", "reconstruction_mse", "reconstruction_mae"):
            points = [diagnostics.get(update, {}).get(metric) for update in (0, 100, 300)]
            rendered = [f"{point:.9g}" if point is not None else "NOT_RECORDED" for point in points]
            change = f"{points[2]-points[1]:.9g}" if all(point is not None for point in points[1:]) else "NOT_AVAILABLE"
            lines.append(f"{value['run_id']:<35} {metric:<22} {rendered[0]:<15} {rendered[1]:<15} {rendered[2]:<15} {change}")
    lines.append("```")
    if not training:
        lines.append("고정 TRAIN 진단 receipt가 아직 없다.")
    lines += ["",
              "## 비용과 자원", "",
              f"마지막 기록 종료까지 wall {resources['wall_seconds_to_last_recorded_end']:.3f}초, 원장에 확정된 GPU 활성 wall {resources['GPU_seconds_committed']:.3f}초. 모델 적재·창 슬라이스·특징 캐시 생성과 timed whole-route 범위를 구분한다.", "",
              f"완료한 최종 adaptation fit {resources['new_adaptation_fits']}개와 버린 TRAIN smoke fit {resources['discarded_TRAIN_smoke_fits']}개를 구분한다. 신규 GT reconstruction/TS2 대규모 사전학습은 0개다. smoke 비용도 GPU 원장에 포함한다.", "",
              "메모리 수치는 시간–오차 그림과 분리한다. 전체 원정밀도는 [COST.csv](COST.csv), 개별 측정은 [COST_TRIALS.csv](COST_TRIALS.csv)에 있다.", "",
              "비용의 p50·IQR·p95는 고정 원점 그룹 3개를 순환한 20회 반복의 경험적 요약이다. 서비스 부하나 독립 요청의 tail 분포를 표집한 실험이 아니므로 서비스 tail 지연 보장으로 해석하지 않는다.", "", "```text", "run_id                                    B   allocated_MiB   reserved_MiB"]
    for row in costs:
        allocated = row.get("peak_allocated_MiB", (row.get("peak_allocated_bytes") or 0) / 1024**2)
        reserved = row.get("peak_reserved_MiB", (row.get("peak_reserved_bytes") or 0) / 1024**2)
        lines.append(f"{row['run_id']:<41} {row['batch']:>1}   {allocated:>13.3f}   {reserved:>12.3f}")
    lines += ["```", "", "## 주장 범위와 검증 한계", "",
              "[확인] Jena 한 데이터·두 seed의 반복 개발 VAL 결과다. 기존 부모의 VAL 선택과 과거 TEST 노출 이력을 제거한 독립 평가가 아니다. 이번 보호 TEST의 읽기·해시·요약은 수행하지 않는다. 사전학습 overlap은 UNKNOWN이다. 두 seed의 차이·표준편차를 일반화 신뢰구간으로 해석하지 않는다.", "",
              "[확인] historical raw_physical SHA 불일치는 미해결 상태로 남긴다. 현재 unchanged loader와 runtime raw 배열 및 기존 6개 모델 좌표 지문·계약의 일치를 별도로 기록한다. 이 차이의 원인을 해결했다고 주장하지 않는다.", "",
              "[확인] GT 경로는 시간축 L→L/4로 4배 축소하지만 latent feature가 2개이므로 scalar 수는 입력 LC에서 2C(L/4)=LC/2로 2배 압축된다. 원채널 C=21을 유지하며 F 시계열은 2C=42개다. TS2 단독은 320개이며 결합은 42개를 유지하지만 encoder 상주·연산이 추가된다. B2의 C→K 채널 압축이나 더 가벼운 전체 시스템을 입증하지 않는다. 실제 F context가 잘렸다면 요청 길이와 실제 길이를 COST_TRIALS에서 함께 확인한다.", "",
              "겹치는 원점은 독립 표본이 아니다. origin별 개선율은 기술 통계이며 사후 통과 기준으로 쓰지 않는다. 학습 미니배치 곡선과 고정 TRAIN 진단점을 구분하고, 과거 재구성은 보조 진단으로만 본다. 정상 실행 후 미개선과 검증 실패·미완료를 JOB_STATUS에서 구분한다.", "",
              *(["[OBSERVATION_CONTEXT_STATS.csv](OBSERVATION_CONTEXT_STATS.csv)의 fully observed subset은 비교 양쪽의 물리 과거 모든 cell이 관측된 원점을 관측 mask만으로 정한 보조 기술 통계다. Phase A 이후에 기준을 등록한 사후 기술 진단이며, 학습·점수 선택 기준이 아니다. 전체 371개 VAL 원점의 주 결과를 교체하지 않는다.", ""] if observation_context_stats else []),
              "## 재현 가능한 산출물", "",
              "- [고정 모델 길이 비교](FIG1_Lookback.png) · [SVG](FIG1_Lookback.svg) · [source CSV](FIG1_Lookback_source.csv)",
              "- [예측 추가학습과 대응 차이](FIG2_Adaptation.png) · [SVG](FIG2_Adaptation.svg) · [source CSV](FIG2_Adaptation_source.csv)",
              "- [전체 지연시간–오차](FIG3_Cost.png) · [SVG](FIG3_Cost.svg) · [source CSV](FIG3_Cost_source.csv)",
              "- [RUNS](RUNS.csv), [COMPARISONS](COMPARISONS.csv), [채널·horizon 충분통계](CHANNEL_HORIZON_STATS.csv), [원점별 효과](ORIGIN_EFFECTS.csv), [학습 곡선](TRAINING_CURVES.csv)",
              "- [방법 변경과 원문](METHOD_DELTA_KO.md), [자원 원장](RESOURCE_LEDGER.json), [작업 상태](JOB_STATUS.csv), [재개 큐](RESUME_QUEUE.md)", "",
              "```bash", "python -m research.gt_ts2_fastcheck_20261018_v1.run report", "python -m research.gt_ts2_fastcheck_20261018_v1.run verify", "```", "",
              "공개 CSV·그림은 receipt에 묶인 private prediction에서 다시 계산한다. 원자료·prediction 배열·pretrained weights·로컬 절대경로는 공개하지 않는다. 게시 commit와 원격 bytes 검증은 별도 publication proof를 따른다.", ""]
    return "\n".join(lines).encode("utf-8")


METHOD = """# 원 방법에서 가져온 부분과 이번에 새로 검증한 부분

[Graph-Time Convolutional Autoencoders, LoG 2022](https://proceedings.mlr.press/v198/sabbaqi22a.html)는 product graph 위의 시공간 convolution, encoder의 시간 downsampling과 decoder의 upsampling을 제안한다. 원 논문은 네트워크 시계열의 비지도 표현과 denoising·anomaly detection을 검증했다. 이번 저장소의 GT는 논문 기반 재구현이며, 기존 reconstruction+sparsity 사전학습 부모를 그대로 출발점으로 쓴다.

[TS2Vec, AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20881)는 증강한 context view의 계층적 대조학습으로 시점별 문맥 표현을 학습한다. 이번에는 기존 Jena·같은 seed의 canonical d320 encoder를 고정하여 알려진 과거 전체에서 offline 표현을 추출한다. 이 경로를 내부 시점별 prefix-causal이라고 부르지 않는다. 예측 원점 이후의 실제 미래는 encoder에 넣지 않는다.

이번 신규 설계는 GT latent와 4시점 평균 TS2 표현을 정렬하고, TRAIN에서만 고정한 feature 통계로 표준화한 뒤 bias 없는 W(320→42)를 거쳐 residual로 더하는 것이다. W는 0에서 시작한다. 4 arm은 기존 GT E/D·공유 product 계수와 필요한 W만 forecast-only MSE로 짧게 적응한다. TS2·F·공간 인접 행렬은 고정한다. RAND는 같은 구조·용량의 무학습 encoder 대조이며 새 대조학습을 하지 않는다.

[AdaPTS, ICML 2025](https://proceedings.mlr.press/v267/benechehab25a.html)의 latent feature-space adapter 뒤에 단변량 foundation model을 적용하는 관점과 연결된다. 그러나 이 논문이 이번 GT/TS2 residual bridge의 효용을 검증했다는 뜻은 아니다.

긴 lookback의 고정 추론, 예측 목적의 공동 추가학습, GT/TS2 잠재 결합, 0.5 출력 평균은 이번 별도 비교다. 신규 GT reconstruction 사전학습이나 GTConvAE/TS2Vec 원 논문 benchmark의 재현으로 표시하지 않는다. GT는 시간축을 L→L/4로 4배 축소하고 feature가 2개라 scalar 수는 LC→LC/2로 2배 압축한다. 원채널 C=21과 F 시계열 42개를 유지하므로 B2의 채널 축소와 구분한다. 300 고정 updates 완료는 수렴 증명이 아니며, CP0·100·300의 고정 TRAIN 진단은 선택에 사용하지 않는 기술 통계다. 성능 미개선도 이 부모·데이터·고정 예산의 결과이며 원 논문 전체나 모든 결합 설계를 기각하지 않는다.
""".encode("utf-8")


def generate(output_dir=None, *, source_report=None, source_cache=None):
    source = Path(source_report) if source_report is not None else c.REPORT
    cache = Path(source_cache) if source_cache is not None else c.CACHE
    output = Path(output_dir) if output_dir is not None else c.REPORT
    contract = _read(source / "EXPERIMENT_CONTRACT.json", {})
    parents = _read(source / "PARENT_BINDINGS.json", {})
    start = _read(source / "START_STATE.json", {})
    decision = _read(source / "UPDATE_DECISION.json", {})
    snapshot = _read(source / "FINAL_RESOURCE_SNAPSHOT.json", {})
    ledger = c.read_json(cache / "GPU_LEDGER.json") if (cache / "GPU_LEDGER.json").exists() else {}
    receipts, bundles, runs = _predictions(source, cache)
    run_index = {row["run_id"]: row for row in runs}
    expected = _expected()
    jobs = [{**spec, "status": receipts.get(spec["run_id"], {}).get("status", "NOT_RUN_MISSING_RECEIPT")}
            for spec in expected]
    comparisons, origin_rows = _paired_rows(receipts, bundles)
    channel_rows = []
    for run_id, bundle in sorted(bundles.items()):
        channels, origins = a.statistics(bundle["prediction"], bundle["target"], bundle["mask"], bundle["origins"], run_id)
        channel_rows.extend(channels)
        origin_rows.extend({"row_type": "RUN", **row} for row in origins)
    training, curves = _training(source, receipts, decision)
    trained = {row["run_id"] for row in training}
    jobs.extend(dict(run_id=spec["run_id"], category="TRAINING",
                     status="COMPLETE" if spec["run_id"] in trained else "NOT_RUN_TRAINING_RECEIPT")
                for spec in expected if spec["phase"] == "B")
    costs, cost_trials = _costs(source, run_index)
    _join_resident(runs, costs)
    smoke = [_read(path) for path in sorted((source / "smoke").glob("*.json"))]
    cost_candidates = contract.get("candidates", [row for row in expected if row["phase"] in ("A", "B")])
    cost_batches = contract.get("cost", {}).get("batches", [1, 8])
    expected_cost = {(spec["run_id"], batch) for spec in cost_candidates if spec.get("phase") in ("A", "B") for batch in cost_batches}
    finished_cost = {(row["run_id"], row["batch"]) for row in costs}
    for run_id, batch in sorted(expected_cost):
        jobs.append(dict(run_id=run_id, category="COST", batch=batch,
                         status="COMPLETE" if (run_id, batch) in finished_cost else "NOT_RUN_COST"))
    for path in sorted((source / "stage_receipts").glob("*.json")):
        stage = _read(path)
        jobs.append(dict(run_id=path.stem, category="STAGE", **stage))
    stages = ledger.get("stages", [])
    ended = max([start.get("started_unix", 0)] + [stage.get("ended_unix", start.get("started_unix", 0)) for stage in stages]
                + [row.get("ended_unix", start.get("started_unix", 0)) for row in jobs if row.get("category") == "STAGE"])
    resources = dict(
        GPU_seconds_committed=ledger.get("GPU_seconds", 0.), GPU_status=ledger.get("status", "UNKNOWN"),
        wall_seconds_to_last_recorded_end=max(0., ended - start.get("started_unix", ended)),
        resource_scope="Committed GPU ledger and wall through the last recorded stage end; replay wall is excluded",
        new_adaptation_fits=len(training), new_pretraining_fits=0, reused_canonical_parents=sum(len(row) for row in parents.values()),
        discarded_TRAIN_smoke_fits=sum(row.get("discarded") is True for row in smoke),
        completed_prediction_receipts=len(runs), expected_prediction_receipts=len(expected),
        completed_cost_receipts=len(costs), expected_cost_receipts=len(expected_cost),
        private_prediction_bytes=sum((cache / "predictions" / (run_id + ".npz")).stat().st_size for run_id in bundles),
        fixed_resource_snapshot=snapshot, GPU_stages=stages,
        finished_training=[{key: value.get(key) for key in ("run_id", "seed", "updates", "seen_origin_count", "unique_seen_origins", "observed_target_exposure", "wall_seconds")} for value in training],
        protected_TEST_accessed=False, publication="Separate publication proof; no self-referential commit manifest")
    status = "COMPLETE" if (len(runs) == len(expected) and len(training) == 8 and expected_cost <= finished_cost
                             and ledger.get("status") == "IDLE" and all(row["status"] == "COMPLETE" for row in jobs)) else "PARTIAL"
    pending = [row for row in jobs if row["status"] != "COMPLETE"]
    resume = ["# 실제 미완료 작업과 재개 큐", "", f"상태: {status}. 기존 완료 receipt·실패 attempt는 덮어쓰지 않는다.", ""]
    if pending:
        resume += [f"- {row['run_id']}" + (f" batch={row['batch']}" if "batch" in row else "") + f": {row['status']}" for row in pending]
        resume += ["", "실패 원인과 부모/데이터 계약을 확인하고 신규 attempt 또는 실제 미완료 단계만 재개한다. 이미 존재하는 sealed stage를 전체 재실행하지 않는다."]
    else:
        resume.append("계획된 대응 실행·비용 측정에 미완료 항목이 없다. 추가 데이터/TEST/예산 확장은 등록하지 않았다.")
    resume += ["", "```bash", "python -m research.gt_ts2_fastcheck_20261018_v1.run report", "python -m research.gt_ts2_fastcheck_20261018_v1.run verify", "```", ""]
    outputs = {
        "RUNS.csv": _csv(runs, ("run_id", "phase", "arm", "seed", "L", "updates", "MSE", "MAE")),
        "COMPARISONS.csv": _csv(comparisons, ("comparison", "seed", "candidate", "reference")),
        "CHANNEL_HORIZON_STATS.csv": _csv(channel_rows, ("run_id", "channel", "horizon_band", "sse", "sae", "count", "mse", "mae")),
        "ORIGIN_EFFECTS.csv": _csv(origin_rows, ("row_type", "run_id", "origin")),
        "COST.csv": _csv(costs, ("run_id", "batch", "MSE", "MAE")),
        "COST_TRIALS.csv": _csv(cost_trials, ("run_id", "batch")),
        "TRAINING_CURVES.csv": _csv(curves, ("run_id", "seed", "row_type", "update")),
        "JOB_STATUS.csv": _csv(jobs, ("run_id", "category", "status")),
        "RESOURCE_LEDGER.json": c.json_bytes(resources), "RESUME_QUEUE.md": "\n".join(resume).encode("utf-8"),
        "README_KO.md": _readme(status, runs, comparisons, costs, training, decision, resources,
                                observation_context_stats=(source / "OBSERVATION_CONTEXT_STATS.csv").is_file()),
        "METHOD_DELTA_KO.md": METHOD,
        **_figure_bytes(runs, comparisons, costs),
    }
    for name, data in outputs.items():
        path = output / name
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError("Preserve existing derived output; use a fresh replay directory: " + name)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        path = output / name
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(data)
    return dict(status=status, verified_prediction_receipts=len(runs), comparisons=len(comparisons),
                metric_parity="FP64_RECOMPUTED_PASS", files={name: c.file_sha(output / name) for name in sorted(outputs)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("report",))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--source-cache", type=Path)
    args = parser.parse_args(argv)
    print(c.json_bytes(generate(args.output_dir, source_report=args.source_report, source_cache=args.source_cache)).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
