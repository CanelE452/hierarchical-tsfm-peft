"""Render sealed four-arm tables and figures without inference or resampling."""

import csv
import json
import math
from pathlib import Path

from runtime import DATASETS, HERE, Job, artifact, check_seal, read_json, save_json


ARMS = ("BOLT_SMALL_LEVEL", "BOLT_SMALL_F0", "BOLT_MINI_F0", "BOLT_TINY_F0")
ALTERNATIVES = ARMS[2:]
PERIODS = ("test_a", "test_b", "combined")
LABELS = dict(zip(ARMS, ("Small + LEVEL", "Small F0", "Mini F0", "Tiny F0")))
DATASET_LABELS = dict(zip(DATASETS, ("Robin", "Peacock Education", "Jena")))
COLORS = dict(zip(ARMS, ("#0072B2", "#E69F00", "#009E73", "#CC79A7")))
MARKERS = dict(zip(ARMS, ("o", "s", "^", "D")))
MIB = 1024 ** 2
REJECTED = "CHANNEL_COMPRESSION_DEPLOYMENT_ADVANTAGE_REJECTED_FOR_TESTED_SETTING"
TRADEOFF = "PARETO_TRADEOFF_REMAINS"
SUPPORTED = "CHANNEL_COMPRESSION_ALLOCATION_SUPPORTED_IN_TESTED_SETTING"
UNRESOLVED = "NO_SINGLE_DOMINANT_ALLOCATION"


def csv_rows(name):
    with (HERE / name).open(encoding="utf-8", newline="") as stream:
        result = []
        for index, raw in enumerate(csv.DictReader(stream)):
            row = {}
            for key, value in raw.items():
                if value == "":
                    row[key] = None
                else:
                    try:
                        row[key] = json.loads(value)
                    except (ValueError, TypeError):
                        row[key] = value
            row["_csv_record"] = index
            row["_csv_line"] = index + 2
            if name == "cost_comparison.csv":
                row["aggregation"] = row.get("row_type", row.get("aggregation"))
                if row["aggregation"] == "block":
                    row["milliseconds_per_origin_median"] = row.get("block_median_ms", row.get("milliseconds_per_origin_median"))
                    row["origins_per_second_median"] = row.get("block_median_origins_s", row.get("origins_per_second_median"))
            result.append(row)
    return result


def num(value, digits=6):
    return "N/A" if value is None else f"{value:.{digits}f}"


def interval_text(values, digits=2):
    return "N/A" if values is None else f"[{num(values[0], digits)}, {num(values[1], digits)}]"


def relative(left, right):
    return None if left is None or right is None or right == 0 else 100 * (left / right - 1)


def english_table(headers, rows):
    return ("| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers)
            + " |\n" + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in rows) + "\n")


def text_table(headers, rows):
    values = [[str(v) for v in row] for row in [headers, *rows]]
    widths = [max(len(row[i]) for row in values) for i in range(len(headers))]
    return "```text\n" + "\n".join("  ".join(v.ljust(widths[i]) for i, v in enumerate(row))
                                   for row in values) + "\n```\n"


def one(rows, **keys):
    matches = [r for r in rows if all(r.get(k) == v for k, v in keys.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one sealed row for {keys}, got {len(matches)}")
    return matches[0]


def score(data, dataset, arm, period="combined"):
    return one(data["accuracy"], dataset=dataset, method=arm, period=period)


def cost(data, dataset, arm, batch=4):
    return one(data["cost"], dataset=dataset, method=arm, batch=batch)


def pair(data, dataset, other, period="combined", metric="mse"):
    return one(data["pairs"], dataset=dataset, left=ARMS[0], right=other, period=period, metric=metric)


def backbone_receipts(preflight):
    result = {}
    for entry in preflight["bolt_models"]:
        result[entry["model"]] = entry["receipt"]
    if set(result) != {"SMALL", "MINI", "TINY"}:
        raise ValueError("Actual loaded receipts for all three backbones are required")
    return result


def load_sources():
    check_seal()
    preflight = read_json(HERE / "preflight_checks.json")
    evaluation = read_json(HERE / "evaluation_summary.json")
    raw_file = read_json(HERE / "cost_cuda_rows.json")
    summary = read_json(HERE / "cost_cuda_summary.json")
    if preflight["status"] != "pass" or evaluation["status"] != "complete":
        raise ValueError("Passed preflight and complete fixed evaluation are required before reporting")
    if raw_file["status"] != "complete" or summary["status"] != "complete":
        raise ValueError("The same-campaign cost receipts must be complete before reporting")
    if len(raw_file["rows"]) != 108 or len(summary["groups"]) != 24:
        raise ValueError("Expected 90 primary blocks, 18 sentinels, and 24 cost groups")
    all_accuracy = csv_rows("accuracy_comparison.csv")
    accuracy = [r for r in all_accuracy if r["aggregation"] != "individual"]
    all_cost = csv_rows("cost_comparison.csv")
    costs = [r for r in all_cost if r["aggregation"] == "summary"]
    pairs = csv_rows("paired_comparisons.csv")
    if len(accuracy) != 36 or len(costs) != 24 or len(pairs) != 108:
        raise ValueError("Fixed table coverage differs from 4 arms x 3 datasets x 3 periods")
    if sum(r["aggregation"] == "block" for r in all_cost) != 108:
        raise ValueError("Cost CSV must preserve every primary and sentinel block")
    result = {"accuracy": accuracy, "cost": costs, "pairs": pairs, "all_cost": all_cost,
              "raw": raw_file["rows"], "summary": summary, "evaluation": evaluation,
              "preflight": preflight, "backbones": backbone_receipts(preflight),
              "models": read_json(HERE / "model_manifest.json")["models"],
              "contract": read_json(HERE / "input_contract.json")["units"]}
    for dataset in DATASETS:
        for arm in ARMS:
            for period in PERIODS:
                row = score(result, dataset, arm, period)
                for metric in ("mse", "mae", "signed_mean_error"):
                    value = row[metric]
                    if value is None or not math.isfinite(value):
                        raise ValueError("A required fixed accuracy value is unavailable")
                expected = evaluation["units"][dataset]["methods"][arm]["periods"][period]
                if any(abs(row[m] - expected[m]) > 1e-12 for m in ("mse", "mae", "signed_mean_error")):
                    raise ValueError("Accuracy CSV is inconsistent with its sealed summary")
            for batch in (1, 4):
                group = cost(result, dataset, arm, batch)
                expected_ids = evaluation["units"][dataset]["methods"][arm]["ids"]
                if set(group["ids"]) != set(expected_ids) or len(group["ids"]) != len(expected_ids):
                    raise ValueError("Accuracy and cost selected instances differ")
                for identifier in expected_ids:
                    blocks = [r for r in raw_file["rows"] if r["dataset"] == dataset and r["id"] == identifier
                              and r["method"] == arm and r["batch"] == batch and r["sentinel"] is None]
                    if len(blocks) != 3 or {r["block"] for r in blocks} != {0, 1, 2}:
                        raise ValueError("All three raw measurement blocks must be present")
                for key in ("deployed_parameters", "parameter_bytes", "buffer_bytes",
                            "deployment_tensor_bytes", "original_fitted_parameter_count",
                            "inference_trainable_parameter_count"):
                    if group.get(key) is None or isinstance(group[key], list):
                        raise ValueError("Actual, consistent deployed state is required for fairness: " + key)
                if group["inference_trainable_parameter_count"] != 0:
                    raise ValueError("Inference must have no trainable parameters")
                if group["deployment_tensor_bytes"] != group["parameter_bytes"] + group["buffer_bytes"]:
                    raise ValueError("Deployed tensor bytes do not reconcile")
            for other in ALTERNATIVES:
                for period in PERIODS:
                    for metric in ("mse", "mae"):
                        p = pair(result, dataset, other, period, metric)
                        if p["absolute_ci95"] is None:
                            raise ValueError("Existing paired intervals are required; reporting never resamples")
    if len(summary["sentinels"]) != 9:
        raise ValueError("All nine dataset/block pre/post sentinel pairs are required")
    return result


def fairness_report(data):
    lines = ["# Comparison fairness\n",
             "[확인] 아래 값은 봉인된 자료·모델 manifest, 사전 실제 로드 receipt와 이번 비용 receipt에서 읽었습니다. "
             "추론 중 gradient를 사용하는 변수 수는 모든 arm에서 0이며, target-data fit 변수 수는 기존 LEVEL의 G 적합 이력입니다. 이번 회차 새 fit은 0입니다.\n",
             "Arm names: `BOLT_SMALL_LEVEL`, `BOLT_SMALL_F0`, `BOLT_MINI_F0`, `BOLT_TINY_F0`.\n"]
    for dataset in DATASETS:
        u = data["contract"][dataset]
        groups = [cost(data, dataset, arm) for arm in ARMS]
        keys = ("SMALL", "SMALL", "MINI", "TINY")
        if (u["l"], u["h"]) != (512, 48):
            raise ValueError("The sealed L/H contract changed")
        for key, group in zip(keys, groups):
            receipt = data["backbones"][key]
            if receipt["family"] != "Chronos-Bolt" or receipt["trainable_parameter_count"] != 0:
                raise ValueError("Loaded family or frozen backbone state cannot be certified")
            if group["deployed_parameters"] < receipt["parameter_count"]:
                raise ValueError("Deployed model has fewer parameters than its declared backbone")
            if receipt["quantiles"][receipt["median_index"]] != 0.5:
                raise ValueError("The reported point forecast is not the native q=0.5")
        for arm, group in zip(ARMS[:2], groups[:2]):
            original = "LEVEL" if arm == ARMS[0] else "F0"
            receipts = [entry["receipt"] for entry in data["preflight"]["baselines"]
                        if entry["receipt"]["dataset"] == dataset and entry["receipt"]["method"] == original]
            if len(receipts) != (2 if original == "LEVEL" else 1):
                raise ValueError("The original selected deployment receipts are incomplete")
            for receipt in receipts:
                for source_key, target_key in (("parameter_count", "deployed_parameters"),
                                               ("parameter_bytes", "parameter_bytes"),
                                               ("buffer_bytes", "buffer_bytes"),
                                               ("original_fitted_parameter_count", "original_fitted_parameter_count")):
                    if receipt[source_key] != group[target_key]:
                        raise ValueError("Preflight and cost deployment state differ")
        rows = [
            ["Family", *("Chronos-Bolt" for _ in ARMS)],
            ["Architecture", *("ChronosBoltModelForForecasting" for _ in ARMS)],
            ["Pinned model revision", *[data["models"][k]["revision"] for k in keys]],
            ["Actual backbone params", *[data["backbones"][k]["parameter_count"] for k in keys]],
            ["Actual backbone parameter bytes", *[data["backbones"][k]["parameter_bytes"] for k in keys]],
            ["Total deployed params", *[g["deployed_parameters"] for g in groups]],
            ["Deployed parameter bytes", *[g["parameter_bytes"] for g in groups]],
            ["Deployed buffer bytes", *[g["buffer_bytes"] for g in groups]],
            ["Total deployed tensor bytes", *[g["deployment_tensor_bytes"] for g in groups]],
            ["Previously fitted target-data params", *[g["original_fitted_parameter_count"] for g in groups]],
            ["Inference gradient-enabled params", *[g["inference_trainable_parameter_count"] for g in groups]],
            ["Original input channels C", *[u["c"] for _ in ARMS]],
            ["TSFM rows per origin", u["k"], *[u["c"] for _ in ARMS[1:]]],
            ["TRAIN PCA", "Fixed C-to-K E/D", "None", "None", "None"],
            ["Residual G", "Previously selected, frozen", "None", "None", "None"],
            ["Target-data fitting history", "TRAIN PCA and selected G", "None", "None", "None"],
            ["New fits in this campaign", 0, 0, 0, 0],
            ["Context / horizon observations", *["512 / 48" for _ in ARMS]],
            ["Precision / TF32", *["FP32 / off" for _ in ARMS]],
            ["Same preprocessing", *("Yes: existing TRAIN statistics and input imputation" for _ in ARMS)],
            ["Same origins / target mask", *("Yes: sealed origin order and raw-finite target mask" for _ in ARMS)],
            ["Point forecast", "Native q=0.5 -> fixed decode + G", *("Native q=0.5, first H48" for _ in ARMS[1:])],
            ["Loss aggregation", "Mean losses of 2 selected seeds", *("One deterministic model" for _ in ARMS[1:])],
            ["TEST exposure", *("Previously exposed; post-exposure practical control" for _ in ARMS)],
            ["Pretraining overlap", *("Unverified" for _ in ARMS)],
        ]
        lines += [f"## {DATASET_LABELS[dataset]} — C{u['c']}/K{u['k']}\n",
                  english_table(["Attribute", *[LABELS[a] for a in ARMS]], rows)]
    lines += ["[확인] Mini/Tiny는 native multivariate 모델이 아니라 같은 Chronos-Bolt 계열의 더 작은 backbone입니다. "
              "모든 F0는 `[B,512,C] -> [B*C,512]`로 원채널을 서로 독립적인 단변량 task로 처리합니다. "
              "LEVEL만 fixed TRAIN PCA 뒤 K개 task를 처리하고 원채널로 decode한 뒤 G를 더합니다. "
              "부모 backbone과 배포 전체 크기를 혼동하지 않으며, 고정 E/D와 G도 배포 tensor bytes에 포함됩니다.\n",
              "이 표는 `model_manifest.json`, `input_contract.json`, `preflight_checks.json`, "
              "`cost_comparison.csv` 및 `cost_cuda_summary.json`에 연결됩니다. 최종 검산·게시 상태는 `final_checks.json`과 `STATUS.md`에서 별도로 확인합니다.\n"]
    return "\n".join(lines)


def overlap(left, right):
    return left is None or right is None or max(left[0], right[0]) <= min(left[1], right[1])


def decide(data, dataset, other):
    p = pair(data, dataset, other)
    l, r = cost(data, dataset, ARMS[0]), cost(data, dataset, other)
    lo, hi = p["absolute_ci95"]
    level_accuracy = hi < 0
    other_accuracy = lo > 0
    keys = ("mean_seed_median_peak_allocated_bytes", "mean_seed_median_origins_s")
    missing = any(x.get(k) is None for x in (l, r) for k in keys)
    ranges_overlap = overlap(l.get("block_median_origins_s_range"), r.get("block_median_origins_s_range"))
    if missing:
        token, case = UNRESOLVED, 4
        reason = "B4 비용 축이 미완료 또는 OOM이므로 단일 allocation 우위를 판정할 수 없습니다."
    else:
        other_cheaper = r[keys[0]] < l[keys[0]] and r[keys[1]] > l[keys[1]]
        level_cheaper = l[keys[0]] < r[keys[0]] and l[keys[1]] > r[keys[1]]
        if lo <= 0 <= hi:
            token, case = UNRESOLVED, 4
            reason = "대응 MSE 구간이 0을 포함하므로 정확도 차이는 불확실합니다. 동등성의 증거가 아닙니다."
        elif ranges_overlap:
            token, case = UNRESOLVED, 4
            reason = "B4 처리량 블록 범위가 겹쳐 안정적인 비용 순위를 확정하지 않습니다. 범위는 신뢰구간이 아닙니다."
        elif other_accuracy and other_cheaper:
            token, case = REJECTED, 1
            reason = f"{LABELS[other]}가 MSE·B4 peak allocated·B4 처리량의 세 주축에서 유리합니다. 이 조건의 압축 deployment 선택 주장을 축소합니다."
        elif level_accuracy and other_cheaper:
            token, case = TRADEOFF, 2
            reason = f"LEVEL은 MSE가 낮지만 {LABELS[other]}는 B4 allocated가 낮고 처리량이 높습니다. 정확도와 비용의 절충이 남습니다."
        elif level_accuracy and level_cheaper:
            token, case = SUPPORTED, 3
            reason = "LEVEL이 MSE·B4 peak allocated·B4 처리량의 세 주축에서 유리합니다. 이 자료·고정 조건의 allocation만 지지합니다."
        else:
            token, case = UNRESOLVED, 4
            reason = "정확도와 주요 비용 축의 방향이 엇갈립니다. 필요한 지표에 따른 조건부 선택만 보고합니다."
    exceptions = []
    if not missing:
        primary_winner = other if case == 1 else ARMS[0] if case == 3 else None
        if primary_winner:
            winner = cost(data, dataset, primary_winner)
            loser_arm = ARMS[0] if primary_winner == other else other
            loser = cost(data, dataset, loser_arm)
            if winner["mean_seed_median_peak_reserved_bytes"] > loser["mean_seed_median_peak_reserved_bytes"]:
                exceptions.append("B4 peak reserved")
            if winner["deployment_tensor_bytes"] > loser["deployment_tensor_bytes"]:
                exceptions.append("deployed tensor bytes")
            w1, l1 = cost(data, dataset, primary_winner, 1), cost(data, dataset, loser_arm, 1)
            if w1["mean_seed_median_ms"] is not None and l1["mean_seed_median_ms"] is not None and w1["mean_seed_median_ms"] > l1["mean_seed_median_ms"]:
                exceptions.append("B1 latency")
            if score(data, dataset, primary_winner)["mae"] > score(data, dataset, loser_arm)["mae"]:
                exceptions.append("combined MAE")
    if exceptions:
        reason += " 다른 축에서 예외가 있습니다: " + ", ".join(exceptions) + ". 모든 지표의 지배로 해석하지 않습니다."
    return {"dataset": dataset, "left": ARMS[0], "right": other, "case": case,
            "token": token, "reason_ko": reason, "accuracy_interval_includes_zero": lo <= 0 <= hi,
            "throughput_block_ranges_overlap": ranges_overlap, "other_axis_exceptions": exceptions,
            "decision_axes": ["combined MSE", "B4 peak allocated", "B4 throughput"],
            "scope": "Fixed exposed datasets and checkpoints; descriptive same-campaign costs"}


def point_frontier(data, dataset):
    result = []
    for arm in ARMS:
        c = cost(data, dataset, arm)
        axes = [score(data, dataset, arm)["mse"], c["mean_seed_median_peak_allocated_bytes"],
                c["mean_seed_median_origins_s"]]
        if any(v is None for v in axes):
            result.append({"arm": arm, "point_dominators": None, "unavailable_reason": "B4 cost unavailable/OOM"})
            continue
        dominators = []
        for other in ARMS:
            if arm == other:
                continue
            co = cost(data, dataset, other)
            values = [score(data, dataset, other)["mse"], co["mean_seed_median_peak_allocated_bytes"],
                      co["mean_seed_median_origins_s"]]
            if any(v is None for v in values):
                continue
            weak = values[0] <= axes[0] and values[1] <= axes[1] and values[2] >= axes[2]
            strict = values[0] < axes[0] or values[1] < axes[1] or values[2] > axes[2]
            if weak and strict:
                dominators.append(other)
        result.append({"arm": arm, "point_dominators": dominators, "mse": axes[0],
                       "peak_allocated_bytes": axes[1], "origins_per_second": axes[2]})
    return result


def cost_display(data, dataset, batch):
    rows = []
    for arm in ARMS:
        c = cost(data, dataset, arm, batch)
        mib = lambda key: None if c.get(key) is None else c[key] / MIB
        resident = c.get("resident_allocated_bytes_range")
        resident_reserved = c.get("resident_reserved_bytes_range")
        rows.append([LABELS[arm], num(c["mean_seed_median_ms"], 3), num(c["mean_seed_median_origins_s"], 3),
                     num(mib("mean_seed_median_peak_allocated_bytes"), 3),
                     num(mib("mean_seed_median_peak_reserved_bytes"), 3),
                     interval_text(c.get("block_median_ms_range"), 3),
                     interval_text(c.get("block_median_origins_s_range"), 3),
                     interval_text(None if resident is None else [v / MIB for v in resident], 3),
                     interval_text(None if resident_reserved is None else [v / MIB for v in resident_reserved], 3)])
    return rows


def claims(decisions):
    rejected = [d for d in decisions if d["token"] == REJECTED]
    supported = [d for d in decisions if d["token"] == SUPPORTED]
    tradeoffs = [d for d in decisions if d["token"] == TRADEOFF]
    unresolved = [d for d in decisions if d["token"] == UNRESOLVED]
    reduced = "; ".join(f"{DATASET_LABELS[d['dataset']]} vs {LABELS[d['right']]}" for d in rejected) or "이번 주축에서 확정된 Case 1 없음"
    kept = "; ".join(f"{DATASET_LABELS[d['dataset']]} vs {LABELS[d['right']]}" for d in supported + tradeoffs) or "내부 구조 대조의 기존 근거와 이번 자료별 조건부 결과"
    return ("**유지:** 같은 Bolt-small 내 compression-only b 보완과 LEVEL/RAW 등의 기존 구조적 결과는 해당 내부 대조의 범위에 유지합니다. "
            f"이번 allocation에서 남는 근거: {kept}.\n\n"
            f"**축소:** 큰 backbone을 유지하며 채널을 줄이면 작은 원채널 backbone보다 deployment가 유리하다는 주장은 {reduced}에 맞춰 축소합니다. "
            "한 자료의 결과를 세 자료의 평균 총점이나 범용 우위로 확장하지 않습니다.\n\n"
            f"**보류:** {len(unresolved)}개 주 비교는 차이 불확실·비용 축 교차·측정 범위 중첩 등의 이유로 단일 우위를 보류합니다. "
            "선택 불확실성, 사전학습 중복 여부, 독립 재현과 보호된 외부 확인은 이 회차에서 해결하지 않았습니다.\n")


def report_ko(data, fairness, decisions, frontiers):
    lines = ["# Chronos-Bolt backbone allocation 비교\n",
             "질문: 큰 Bolt-small의 처리 series를 C→K로 줄이고 기존 LEVEL로 보정하는 선택은 더 작은 동일 계열 backbone으로 C개 원채널을 그대로 처리하는 선택보다 정확도–추론비용의 선택 이유가 남는가?\n",
             "이번 결과는 LEVEL method ablation이 아닌 **same-family practical allocation comparison**입니다. "
             "backbone size와 channel allocation, 기존 target-data adaptation 이력이 함께 다릅니다. 모든 세 자료는 이미 TEST가 노출된 후속 비교입니다.\n",
             fairness.replace("# Comparison fairness", "## 비교 공정성", 1),
             "## 정확도: TEST-A / TEST-B / combined\n"]
    for dataset in DATASETS:
        rows = [[LABELS[arm], period, num(score(data, dataset, arm, period)["mse"]),
                 num(score(data, dataset, arm, period)["mae"]),
                 num(score(data, dataset, arm, period)["signed_mean_error"]),
                 score(data, dataset, arm, period)["target_count"], score(data, dataset, arm, period)["origins"]]
                for arm in ARMS for period in PERIODS]
        lines += [f"### {DATASET_LABELS[dataset]}\n", text_table(
            ["Arm", "Period", "MSE", "MAE", "Signed bias", "Observed targets", "Origins"], rows)]
    lines += ["[확인] Primary는 관측 target의 TRAIN-standardized channel-macro MSE, secondary는 동일 MAE입니다. "
              "combined는 A/B별 채널 SSE·절대오차합·count를 먼저 합치고 채널별 비율 뒤 채널 평균을 냈습니다. "
              "A/B MSE 단순 평균이나 자료 간 MSE 평균 총점을 만들지 않았습니다. LEVEL은 기존 두 선택 seed의 손실 평균이며 prediction ensemble이 아닙니다. "
              "Small/Mini/Tiny F0는 각각 deterministic model 한 번으로 집계했습니다.\n",
              "## 대응 상대 차이와 조건부 구간\n",
              "상대 차이는 `100*(LEVEL/alternative - 1)`이며 음수는 LEVEL의 낮은 오차입니다. "
              "기존 paired time-block bootstrap을 재사용한 조건부 95% 구간으로 모델 선택 불확실성은 포함하지 않습니다. "
              "A/B 경계를 넘지 않고 method·origin·channel pairing을 유지했습니다. 구간의 0 포함은 차이 불확실이며 동등성·비열등성의 증거가 아닙니다.\n"]
    for dataset in DATASETS:
        rows = []
        for other in ALTERNATIVES:
            for period in PERIODS:
                for metric in ("mse", "mae"):
                    p = pair(data, dataset, other, period, metric)
                    rows.append([LABELS[other], period, metric.upper(), num(p["relative_percent"], 2),
                                 interval_text(p["relative_ci95_percent"]), num(p["absolute_difference"]),
                                 interval_text(p["absolute_ci95"], 6),
                                 "Includes zero" if p["absolute_ci95"][0] <= 0 <= p["absolute_ci95"][1] else "Excludes zero"])
        lines += [f"### {DATASET_LABELS[dataset]}\n", text_table(
            ["Alternative", "Period", "Metric", "LEVEL relative %", "Conditional 95% %", "LEVEL minus alt", "Conditional 95%", "Difference"], rows)]
    lines += ["보조 Small LEVEL/F0, Small F0/Mini, Small F0/Tiny, Mini/Tiny까지 총6개 대응 비교의 A/B/combined·MSE/MAE는 "
              "`paired_comparisons.csv`에 보존합니다. block length Robin/Peacock7, Jena42 origins; resamples2000; seed9262026; "
              "dataset salt0/0/1과 기존 기간 층화를 유지했습니다. 보고 생성기는 bootstrap을 추가 수행하지 않습니다.\n",
              "## 같은 회차 비용과 배포 상태\n",
              "[확인] RTX 4070, FP32, TF32 off, 동일 환경·CPU4 threads에서 첫24 chronological VAL origins를 사용했습니다. "
              "각 shape 예열10, 전체24-origin passes10, 순서회전3 blocks입니다. 측정은 standardized CPU input에서 reshape/transfer, "
              "전체 online inference, H48C contiguous CPU output 복귀까지입니다. LEVEL의 PCA encode·Bolt-small·decode·G를 모두 포함하고, "
              "load/disk/hash/score/log는 latency에서 제외했습니다. compile·quantization·임의 B2 대안은 없습니다.\n"]
    for dataset in DATASETS:
        for batch in (1, 4):
            lines += [f"### {DATASET_LABELS[dataset]} — B{batch}\n", text_table(
                ["Arm", "ms/origin", "origins/s", "Peak alloc MiB", "Peak reserved MiB", "Block ms range", "Block origins/s range", "Resident alloc range MiB", "Resident reserved range MiB"],
                cost_display(data, dataset, batch))]
    differences = []
    for dataset in DATASETS:
        level4, level1 = cost(data, dataset, ARMS[0]), cost(data, dataset, ARMS[0], 1)
        for other in ALTERNATIVES:
            other4, other1 = cost(data, dataset, other), cost(data, dataset, other, 1)
            differences.append([DATASET_LABELS[dataset], LABELS[other],
                                num(relative(other1["mean_seed_median_ms"], level1["mean_seed_median_ms"]), 2),
                                *[num(relative(other4[k], level4[k]), 2) for k in
                                  ("mean_seed_median_origins_s", "mean_seed_median_peak_allocated_bytes",
                                   "mean_seed_median_peak_reserved_bytes", "deployment_tensor_bytes")]])
    lines += ["### Mini/Tiny의 LEVEL 대비 비용 차이\n",
              "아래 상대 차이는 `100*(alternative/LEVEL - 1)`입니다. 처리량의 양수와 latency·bytes의 음수가 alternative에 유리합니다.\n",
              text_table(["Dataset", "Alternative", "B1 latency %", "B4 throughput %", "B4 allocated %", "B4 reserved %", "Deployed tensor bytes %"], differences)]
    lines += ["집계는 instance별3 block median의 median을 구한 뒤 LEVEL 선택 seed2개의 값을 평균했습니다. "
              "범위는 모든 선택 seed/block 원값의 min–max이며 신뢰구간이 아닙니다. allocated, reserved, "
              "모델 로드·cleanup 직후 resident, deployed tensor bytes는 다른 양입니다. CPU process memory도 원블록에 보존합니다.\n"]
    deployed = [[DATASET_LABELS[ds], LABELS[a], cost(data, ds, a)["deployed_parameters"],
                 cost(data, ds, a)["parameter_bytes"], cost(data, ds, a)["buffer_bytes"],
                 cost(data, ds, a)["deployment_tensor_bytes"], cost(data, ds, a)["original_fitted_parameter_count"]]
                for ds in DATASETS for a in ARMS]
    lines += [text_table(["Dataset", "Arm", "Deploy params", "Parameter bytes", "Buffer bytes", "Total tensor bytes", "Prior fit params"], deployed),
              "Mini/Tiny의 작은 실제 배포 가중치는 배포 선택에서 의미가 있습니다. 추가 fit0은 사전학습 비용·가중치 bytes·resident memory0을 뜻하지 않습니다. "
              "LEVEL의 이전 G 적합과 이번 inference gradient-enabled params0은 별도로 해석합니다.\n",
              "### 모든 원블록과 drift\n"]
    raw = [r for r in data["all_cost"] if r["aggregation"] == "block"]
    lines += [text_table(["Dataset", "Arm", "ID", "Block", "B", "Sentinel", "ms/origin", "origins/s", "Peak alloc MiB", "Peak reserved MiB", "Status"],
                        [[DATASET_LABELS[r["dataset"]], LABELS[r["method"]], r["id"], r["block"], r["batch"],
                          r.get("sentinel") or "primary", num(r.get("milliseconds_per_origin_median"), 3),
                          num(r.get("origins_per_second_median"), 3),
                          num(None if r.get("peak_allocated_bytes") is None else r["peak_allocated_bytes"] / MIB, 3),
                          num(None if r.get("peak_reserved_bytes") is None else r["peak_reserved_bytes"] / MIB, 3),
                          r.get("status") or "see raw receipt"] for r in raw])]
    lines += [text_table(["Dataset", "Block", "F0 pre ms", "F0 post ms", "post/pre-1 %"],
                        [[DATASET_LABELS[s["dataset"]], s["block"], num(s["pre_ms"], 3), num(s["post_ms"], 3),
                          num(None if s["post_relative_change"] is None else 100 * s["post_relative_change"], 2)] for s in data["summary"]["sentinels"]]),
              "이 표와 `cost_cuda_rows.json`/`cost_comparison.csv`는 전체108 block/sentinel과10 pass 원시간, "
              "resident allocated/reserved, CPU process memory 및 telemetry를 보존합니다. drift의 원인은 확정하지 않습니다. "
              "이전 회차 timing과 이번 값을 나누어 속도비를 만들지 않았습니다.\n",
              "## 자료별 Pareto와 Phase 0 판정\n"]
    for dataset in DATASETS:
        lines.append(f"### {DATASET_LABELS[dataset]}\n")
        for d in [v for v in decisions if v["dataset"] == dataset]:
            lines.append(f"- **LEVEL vs {LABELS[d['right']]} — Case {d['case']}:** `{d['token']}`. {d['reason_ko']}\n")
        lines.append(text_table(["Arm", "Point dominators in fixed grid"],
            [[LABELS[p["arm"]], "B4 unavailable/OOM" if p["point_dominators"] is None else
              "; ".join(LABELS[a] for a in p["point_dominators"]) or "None (point nondominated)"] for p in frontiers[dataset]]))
    lines += ["Pareto 표는 MSE·B4 allocated·B4 throughput의 **점추정**만 사용한 기술적 고정 grid입니다. "
              "조건부 구간이0을 포함하거나 비용축이 교차하면 단일 우위를 선언하지 않습니다. "
              "reserved·resident·B1·가중치 bytes·MAE를 포함한 모든 지표의 지배, 모든 K의 전선, 범용 우위를 뜻하지 않습니다. "
              "임의 가중 효율 총점·비열등성 margin은 도입하지 않았습니다.\n",
              "## 가장 강한 반론\n",
              "- **A — 작은 backbone이 싸지는 것은 당연하다:** 맞습니다. 비용 감소의 신규성 대신 그 대가인 정확도와 실제 전체 배포 비용을 함께 비교하는 질문입니다.\n",
              "- **B — LEVEL adaptation과 backbone 선택은 method 비교가 아니다:** 맞습니다. 같은 계열의 practical allocation/system comparison으로 명명하며 순수 LEVEL method ablation이나 압축만의 인과 효과로 쓰지 않습니다.\n",
              "- **C — Mini/Tiny가 이기면 LEVEL 설계가 틀렸는가:** 아닙니다. 같은 Small 안의 COMPRESS→LEVEL, LEVEL/RAW 등 내부 구조 효과는 별도 근거이며 여기서는 deployment 선택 이유만 판정합니다.\n",
              "- **D — Mini/Tiny에도 LEVEL을 붙이면 되는가:** 별도 후속 연구 질문입니다. 이번 primary 네 arm의 답을 얻는 데 필요하지 않아 자동 실행하지 않습니다.\n",
              "## LEVEL 주장과 다음 실험\n", claims(decisions),
              "**다음 실험:** 이번 고정 allocation 질문을 답하기 위해 Mini/Tiny+LEVEL, LoRA, 새 K·G·seed·dataset search는 필요하지 않으며 자동 실행하지 않습니다. "
              "보편적 deployment 주장을 새로 하려면 노출되지 않은 자료·새 장치에서 사전 고정한 별도 확인이 필요합니다. "
              "블록 중첩으로 안정적인 비용 순위를 원하는 경우에는 그 조건만 대상으로 독립 비용 세션을 별도 질문으로 검토할 수 있습니다.\n",
              "## 근거와 실제 상태\n",
              "새 fit0과 optimizer update0이며 Mini/Tiny는2모델×3자료의6 canonical TEST inference units입니다. "
              "실제 실행 수·forward call·실패·재시도는 `prediction_manifest.json`, `evaluation_summary.json`, `ledger.json`의 실제 receipt를 기준으로 확인합니다. "
              "기존 LEVEL/F0 prediction은 hash·shape·origin·mask binding 후 재사용했습니다. "
              "검산 완료·commit/push 상태를 이 보고 생성기가 선행 선언하지 않으며 `final_checks.json`과 `STATUS.md`를 확인합니다. "
              "자체검산은 독립 재현 또는 논문 신규성 인증이 아닙니다.\n",
              "기존 Chronos-2/Chronos-2-Small 비교는 다른 family의 native multivariate system control이며 이번 same-family Bolt allocation과 별도입니다. "
              "원본 PPT/PDF 편집 없이 `PRESENTATION_PATCH.md`에 수정 제안만 남겼습니다. "
              "출력 그림과 CSV hash·사용 행 manifest는 `figures/manifest.json`에 있습니다.\n"]
    return "\n".join(lines)


def presentation_patch(data, decisions):
    lines = ["# 발표 수정 제안: 동일 Chronos-Bolt 계열 allocation\n",
             "이 문서는 원본 PPT/PDF를 편집한 결과가 아닌 역할별 추가 제안입니다. "
             "기존 Chronos-2 native multivariate system comparison과 별도 표·캡션으로 유지합니다.\n",
             "- **질문/동기:** ‘큰 Small을 압축하는 대신 더 작은 Bolt에 원채널을 넣으면?’을 직접 질문으로 추가합니다.\n",
             "- **자료·정보 계약:** Robin C17/K5, Peacock C13/K4, Jena C21/K6, L512/H48와 기존 TRAIN 통계·결측대체·origin·target mask를 명시합니다. F0들은 원채널 독립 단변량이라는 공통점을 표시합니다.\n",
             "- **공정성/파라미터:** `FAIRNESS_TABLE.md`를 정확도 결론 전에 둡니다. G 이전 학습 변수, inference gradient0, backbone params, 전체 배포 params/bytes와 K/C task rows를 구분합니다.\n",
             "- **MSE/MAE 주표:** Small+LEVEL, Small F0 anchor, Mini F0, Tiny F0의4행을 추가하되 세 자료를 평균하지 않습니다. combined 본문과 TEST-A/B 부록을 연결합니다. LEVEL의 두 선택 seed는 손실 평균이며 zero-shot은1개입니다.\n",
             "- **대응 구간:** LEVEL/Mini와 LEVEL/Tiny의 상대 MSE·MAE와 조건부95% 구간을 표시합니다. 0 포함은 ‘차이 불확실’로 쓰고 ‘동등’이라고 바꾸지 않습니다.\n",
             "- **비용:** 이번 같은 RTX4070 세션 B1 latency, B4 throughput, peak allocated/reserved를 사용합니다. 작은 Mini/Tiny의 전체 가중치 bytes도 명시합니다. raw block 범위·resident·sentinel drift는 부록에 두며 범위를 CI라고 부르지 않습니다.\n",
             "- **내부 구조 대조:** compression-only b, LEVEL/RAW, 내부 ablation 결과는 현재 same-family allocation과 별도 근거로 유지합니다. deployment 선택에서 지배된 조건이 구조 효과의 부정이라는 문장을 추가하지 않습니다.\n",
             "## 추가할 combined accuracy table\n"]
    lines.append(english_table(["Arm", "Robin MSE", "Robin MAE", "Peacock MSE", "Peacock MAE", "Jena MSE", "Jena MAE"],
                              [[LABELS[a], *[num(score(data, ds, a)[m]) for ds in DATASETS for m in ("mse", "mae")]] for a in ARMS]))
    lines += ["**English caption:** Same-family practical allocation comparison on previously exposed datasets. "
              "All F0 models process each original channel as an independent univariate task. LEVEL uses a fixed TRAIN PCA, "
              "the pinned Small backbone, and a previously selected frozen G. Errors pool period sums/counts before channel averaging; "
              "LEVEL averages selected-seed losses. This is neither a method ablation nor independent confirmation.\n",
              "## 자료별 본문 교체 문장\n"]
    for d in decisions:
        p = pair(data, d["dataset"], d["right"])
        lines.append(f"- **{DATASET_LABELS[d['dataset']]} / {LABELS[d['right']]}:** LEVEL 상대 MSE {num(p['relative_percent'], 2)}%, "
                     f"조건부95% 구간 {interval_text(p['relative_ci95_percent'])}%. `{d['token']}`. {d['reason_ko']}\n")
    lines += ["## 주장 조정\n", claims(decisions),
              "그림은 `figures/allocation_<dataset>.png`/`.svg`입니다. 각 자료의 두 영어 패널은 combined MSE와 B4 throughput, "
              "combined MSE와 B4 allocated를 보여 줍니다. 축은0을 포함하며 whisker는 observed block range입니다. "
              "원본 PPT/PDF 편집, Mini/Tiny+LEVEL 또는 새 실험 실행은 수행하지 않습니다.\n"]
    return "\n".join(lines)


def make_figures(data, decisions, frontiers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    directory = HERE / "figures"
    directory.mkdir(exist_ok=True)
    sources = {name: artifact(HERE / name) for name in
               ("accuracy_comparison.csv", "paired_comparisons.csv", "cost_comparison.csv")}
    manifest = {"schema": "level_bolt_allocation_figures_v1", "status": "generated_from_complete_tables",
                "sources": sources, "code": artifact(__file__), "no_inference": True,
                "no_resampling": True, "zero_axes": True,
                "whiskers": "Observed all-selected-instance/block range; not a confidence interval",
                "figure_rows": [], "decisions": decisions, "point_frontiers": frontiers}
    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))
        fig.subplots_adjust(left=.075, right=.985, top=.79, bottom=.20, wspace=.25)
        fig.suptitle(DATASET_LABELS[dataset] + ": fixed Chronos-Bolt allocation", y=.97, fontsize=15)
        used = []
        for arm in ARMS:
            a, c = score(data, dataset, arm), cost(data, dataset, arm)
            used.append({"dataset": dataset, "arm": arm,
                         "accuracy_csv_record": a["_csv_record"], "accuracy_csv_line": a["_csv_line"],
                         "cost_csv_record": c["_csv_record"], "cost_csv_line": c["_csv_line"],
                         "period": "combined", "batch": 4, "mse": a["mse"],
                         "origins_s": c["mean_seed_median_origins_s"],
                         "peak_allocated_bytes": c["mean_seed_median_peak_allocated_bytes"]})
            for ax, key, range_key, divisor in (
                    (axes[0], "mean_seed_median_origins_s", "block_median_origins_s_range", 1),
                    (axes[1], "mean_seed_median_peak_allocated_bytes", "allocated_range_bytes", MIB)):
                x = c[key]
                if x is None:
                    continue
                bounds = c.get(range_key)
                x /= divisor
                ax.scatter(x, a["mse"], color=COLORS[arm], marker=MARKERS[arm], s=85,
                           edgecolors="black", linewidths=.6, zorder=4, label=LABELS[arm])
                if bounds is not None:
                    ax.hlines(a["mse"], bounds[0] / divisor, bounds[1] / divisor,
                              color=COLORS[arm], linewidth=1.6, zorder=3)
                    ax.plot([bounds[0] / divisor, bounds[1] / divisor], [a["mse"], a["mse"]],
                            linestyle="none", marker="|", color=COLORS[arm], markersize=8)
        for ax in axes:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Combined channel-macro MSE (lower is better)")
            ax.grid(True, color="#cccccc", linewidth=.6, alpha=.7)
        axes[0].set_title("Accuracy and batch-4 throughput")
        axes[0].set_xlabel("Batch-4 origins/s (higher is better)")
        axes[1].set_title("Accuracy and batch-4 allocated memory")
        axes[1].set_xlabel("Batch-4 peak allocated MiB (lower is better)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .905), ncol=4,
                   frameon=False, columnspacing=1.8)
        foot = ("FP32, TF32 off, RTX 4070; full CPU-input to CPU-output scope.\n"
                "Whiskers: observed block range, not CI. Previously exposed evaluation; fixed models/K.\n"
                "Sources: accuracy_comparison.csv " + sources["accuracy_comparison.csv"]["sha256"][:12]
                + "; cost_comparison.csv " + sources["cost_comparison.csv"]["sha256"][:12])
        fig.text(.5, .035, foot, ha="center", va="bottom", fontsize=8, color="#333333")
        outputs = []
        for suffix in ("png", "svg"):
            path = directory / f"allocation_{dataset}.{suffix}"
            fig.savefig(path, dpi=300, facecolor="white")
            outputs.append(artifact(path))
        plt.close(fig)
        manifest["figure_rows"].append({"dataset": dataset, "outputs": outputs, "plotted_rows": used})
    save_json(directory / "manifest.json", manifest)
    return manifest


def readme():
    return """# Chronos-Bolt same-family allocation control

This campaign compares BOLT_SMALL_LEVEL, BOLT_SMALL_F0, BOLT_MINI_F0, and BOLT_TINY_F0 under the existing Robin/Peacock/Jena L512/H48 contracts. It is a post-exposure practical allocation comparison, not a method ablation or independent confirmation.

- [Question and limits](PURPOSE.md), [fixed plan](PLAN.md), [execution contract](REQUEST.txt), and [actual current status](STATUS.md).
- [Fairness before conclusions](FAIRNESS_TABLE.md), [Korean report](FINAL_REPORT_KO.md), and [presentation suggestions](PRESENTATION_PATCH.md).
- Accuracy: `accuracy_comparison.csv`, `accuracy_channels.csv`, `accuracy_origins.csv`, and all six paired comparisons in `paired_comparisons.csv`.
- Costs: `cost_comparison.csv` retains 24 summaries and all 108 raw blocks/sentinels. `cost_cuda_rows.json` retains pass times, allocated/reserved/resident memory, deployment state, CPU memory, and telemetry. Block ranges are not confidence intervals.
- Provenance: `model_manifest.json`, `input_contract.json`, `reuse_manifest.json`, `preflight_checks.json`, `prediction_manifest.json`, `cost_binding.json`, and `evaluation_seal.json`.
- Figures: `figures/allocation_robin.png`, `figures/allocation_peacock_education.png`, and `figures/allocation_jena.png`, with matching SVG exports. `figures/manifest.json` binds exact CSV record/line keys and source SHA256 hashes.
- Final verification/publication state belongs to `final_checks.json`, `ledger.json`, and `STATUS.md`. Generated reports do not themselves certify PASS, independent reproduction, or a remote commit.

`report_controls.py` only reads completed sealed CSV/JSON artifacts and renders reports/Matplotlib figures. It performs no fitting, model forwards, downloads, or new bootstrap draws. Use the existing project Python environment after evaluation and matched cost tables have completed. Large predictions and checkpoints remain in the campaign cache and are not public artifacts. Original PPT/PDF files are unchanged.
"""


def main():
    with Job("cpu", "generate_result_reports", reserve_s=60,
             metadata={"fit_count": 0, "model_inference": False, "bootstrap_draws": 0}) as job:
        data = load_sources()
        fairness = fairness_report(data)
        decisions = [decide(data, ds, arm) for ds in DATASETS for arm in ALTERNATIVES]
        frontiers = {ds: point_frontier(data, ds) for ds in DATASETS}
        manifest = make_figures(data, decisions, frontiers)
        for name, content in (
                ("FAIRNESS_TABLE.md", fairness),
                ("FINAL_REPORT_KO.md", report_ko(data, fairness, decisions, frontiers)),
                ("PRESENTATION_PATCH.md", presentation_patch(data, decisions)),
                ("README.md", readme())):
            (HERE / name).write_text(content + "\n", encoding="utf-8")
        job.heartbeat({"phase": "reports_and_figures_written", "figures": len(manifest["figure_rows"]),
                       "fit_count": 0, "model_inference": False})
        print(json.dumps({"status": "generated_from_complete_tables", "fit_count": 0,
                          "model_inference": False, "decisions": decisions}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
