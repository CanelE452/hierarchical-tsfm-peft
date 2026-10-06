"""Write the blocked-at-preflight report and figures from saved receipts only."""
import csv
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from runtime_v18 import DATASETS, HERE, ROOT, artifact, read_json, save_json


LABELS = {"robin": "Robin", "peacock_education": "Peacock Education", "jena": "Jena"}
OLD = ROOT / "research/level_bolt_backbone_scaling_v1_20261006"
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#666666"]


def write_text(name, text):
    (HERE / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(name, rows):
    with (HERE / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def historical_level_rows():
    path = OLD / "accuracy_comparison.csv"
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if row["method"] == "BOLT_SMALL_LEVEL" and row["aggregation"] != "individual":
                rows.append({"dataset": row["dataset"], "period": row["period"], "mse": float(row["mse"]),
                             "mae": float(row["mae"]), "signed_mean_error": float(row["signed_mean_error"]),
                             "id": row["id"], "csv_record": index, "csv_line": index + 2,
                             "source_pointer": f"CSV record {index}, physical line {index + 2}",
                             "reuse_scope": "Previously stored mean of two selected seed losses; no new TEST or ED comparison"})
    if len(rows) != 9:
        raise ValueError("Expected nine existing LEVEL period rows")
    return artifact(path), rows


def level_parameter_receipts():
    path = OLD / "preflight_checks.json"
    values = {}
    for index, row in enumerate(read_json(path)["baselines"]):
        if row["method"] == "LEVEL":
            values.setdefault(row["dataset"], []).append({"receipt": row["receipt"], "pointer": f"/baselines/{index}/receipt"})
    if any(len(values.get(ds, [])) != 2 for ds in DATASETS):
        raise ValueError("Expected two historical loaded LEVEL parameter receipts per dataset")
    return artifact(path), values


def prepare_tables(preflight, historical_params):
    summary = []
    for ds in DATASETS:
        row = preflight["units"][ds]
        receipt = row["model_receipt"]
        old = historical_params[ds][0]["receipt"]
        summary.append({"dataset": ds, "C": receipt["channels"], "K": receipt["latent"],
                        "canonical_VAL_origins": row["canonical_VAL_origins"],
                        "max_absolute_error": row["step0_COMPRESS_parity"]["max_absolute"],
                        "absolute_tolerance_component": row["step0_COMPRESS_parity"]["atol"],
                        "relative_tolerance": row["step0_COMPRESS_parity"]["rtol"],
                        "ED_trainable_parameters": receipt["trainable_parameter_count"],
                        "historical_LEVEL_G_parameters": old["original_fitted_parameter_count"],
                        "ED_parameter_bytes": receipt["parameter_bytes"], "ED_buffer_bytes": receipt["buffer_bytes"],
                        "ED_total_tensor_bytes": receipt["total_tensor_bytes"],
                        "historical_LEVEL_parameter_bytes": old["parameter_bytes"],
                        "historical_LEVEL_buffer_bytes": old["buffer_bytes"],
                        "source_json_pointer": f"/units/{ds}", "scope": "Pre-training feasibility; zero new fits/TEST/updates"})
    estimate = preflight["estimate"]
    projection = [{"component": ds + "_four_full_fits", "projected_seconds": estimate["units"][ds]["four_fits_upper_s"],
                   "source_json_pointer": f"/estimate/units/{ds}/four_fits_upper_s", "kind": "Planning projection, not actual fit time"} for ds in DATASETS]
    projection += [{"component": "post_training", "projected_seconds": estimate["post_training_upper_s"],
                    "source_json_pointer": "/estimate/post_training_upper_s", "kind": "TEST, cost, diagnostics and model-load planning allowance"},
                   {"component": "preflight_at_estimate", "projected_seconds": estimate["measured_preflight_gpu_job_s"],
                    "source_json_pointer": "/estimate/measured_preflight_gpu_job_s", "kind": "Elapsed preflight at projection snapshot; terminal ledger elapsed is separate"}]
    write_csv("preflight_summary.csv", summary)
    write_csv("resource_projection.csv", projection)
    return summary, projection


def save_figure(fig, stem):
    outputs = []
    for extension in ("png", "svg"):
        path = HERE / "figures" / (stem + "." + extension)
        fig.savefig(path, dpi=220, facecolor="white")
        outputs.append(artifact(path))
    plt.close(fig)
    return outputs


def figures(summary, projection, preflight, params_source):
    directory = HERE / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "axes.labelsize": 11,
                         "axes.titlesize": 12, "legend.fontsize": 9})
    sources = {n: artifact(HERE / n) for n in ("preflight_summary.csv", "resource_projection.csv", "preflight.json")}
    sources["historical_LEVEL_enumeration"] = params_source
    budget_source = sources["resource_projection.csv"]
    fig, ax = plt.subplots(figsize=(12.5, 4.7))
    fig.subplots_adjust(left=.08, right=.92, top=.80, bottom=.35)
    start = 0.0
    names = ["Robin: 4 full fits", "Peacock: 4 full fits", "Jena: 4 full fits", "Post-training scope", "Preflight snapshot"]
    for row, name, color in zip(projection, names, COLORS):
        hours = float(row["projected_seconds"]) / 3600
        ax.barh(0, hours, left=start, color=color, edgecolor="white", height=.40, label=name)
        if hours > .10:
            ax.text(start + hours / 2, 0, f"{hours:.2f} h", ha="center", va="center", color="white", fontweight="bold")
        start += hours
    ax.axvline(3, color="black", linestyle="--", linewidth=1.5)
    ax.text(3, .30, "Adopted cap: 3 h", ha="right", va="bottom", fontsize=10)
    ax.text(start + .045, 0, f"{start:.3f} h\nSTOP_RESOURCE", va="center", fontsize=10, fontweight="bold")
    ax.set(xlim=(0, start + .38), ylim=(-.32, .50), yticks=[], xlabel="Projected cumulative GPU-job wall time (hours)")
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    fig.legend(loc="center", bbox_to_anchor=(.5, .18), ncol=3, frameon=False)
    fig.suptitle("Resource gate stopped the campaign before training", x=.08, ha="left", fontsize=15, fontweight="bold")
    fig.text(.08, .85, "Planning projection: all 12 fits at 120 epochs; no early stopping assumed. Actual preflight: 21.85 s.", fontsize=10)
    fig.text(.08, .075, "Max of 3 zero-update backward timings, 1.25 factor, whole VAL passes and post-load allowances.\nThis is not measured convergence time or a proof that a 3 h campaign is impossible.", fontsize=9, color="#444444")
    fig.text(.08, .025, "Source: resource_projection.csv | SHA256 " + budget_source["sha256"], fontsize=7.5, family="monospace")
    first = {"id": "resource_projection", "scope": "Stored planning estimate, not a method-cost comparison",
             "csv_source": budget_source, "csv_records": list(range(5)), "csv_physical_lines": list(range(2, 7)),
             "source_json_pointers": [r["source_json_pointer"] for r in projection],
             "outputs": save_figure(fig, "resource_projection")}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    fig.subplots_adjust(left=.075, right=.975, top=.77, bottom=.27, wspace=.32)
    positions = np.arange(3)
    labels = [LABELS[r["dataset"]] + f"\nC={r['C']}, K={r['K']}" for r in summary]
    errors = [float(r["max_absolute_error"]) for r in summary]
    axes[0].bar(positions, errors, color=COLORS[0], width=.50, label="Stored max absolute error")
    axes[0].axhline(1e-5, color="black", linestyle="--", linewidth=1.3, label="Absolute tolerance component")
    axes[0].set(xticks=positions, xticklabels=labels, ylim=(0, 1.23e-5), ylabel="Absolute forecast difference", title="Step-0 ED vs COMPRESS on full VAL")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(-5, -5))
    for i, value in enumerate(errors):
        axes[0].text(i, value + 3.5e-7, f"{value:.2e}", ha="center", fontsize=9)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -.22), frameon=False, fontsize=8.5)
    ed = [int(r["ED_trainable_parameters"]) for r in summary]
    level = [int(r["historical_LEVEL_G_parameters"]) for r in summary]
    axes[1].bar(positions - .17, ed, width=.32, color=COLORS[0], label="ED: initial weights, new enumeration")
    axes[1].bar(positions + .17, level, width=.32, color=COLORS[1], label="LEVEL G: historical loaded enumeration")
    axes[1].set(xticks=positions, xticklabels=labels, ylim=(0, max(level) * 1.15), ylabel="Parameters in the trainable component", title="Actual counts; not parameter matched")
    for i, (small, large) in enumerate(zip(ed, level)):
        axes[1].text(i - .17, small + 500, f"{small:,}", ha="center", fontsize=9)
        axes[1].text(i + .17, large + 500, f"{large:,}", ha="center", fontsize=9)
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, -.22), frameon=False, fontsize=8.5)
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Pre-training readiness | zero fits, zero TEST predictions, zero updates", x=.075, ha="left", fontsize=14, fontweight="bold")
    fig.text(.075, .84, "All three full-VAL parity checks passed atol=1e-5 + rtol=1e-4 | backbone states unchanged", fontsize=10)
    fig.text(.075, .065, "Finite encoder and decoder gradients were observed at microbatch 4 / 2 / 1. These checks do not establish trained accuracy, cost advantage or novelty.", fontsize=8.5, color="#444444")
    fig.text(.075, .023, "Source: preflight_summary.csv | SHA256 " + sources["preflight_summary.csv"]["sha256"], fontsize=7.5, family="monospace")
    second = {"id": "preflight_readiness", "scope": "Stored step0 feasibility and actual parameter enumeration only",
              "csv_source": sources["preflight_summary.csv"], "csv_records": [0, 1, 2], "csv_physical_lines": [2, 3, 4],
              "source_json_pointers": [r["source_json_pointer"] for r in summary],
              "outputs": save_figure(fig, "preflight_readiness")}
    save_json(directory / "manifest.json", {"schema": "matched_ed_blocked_figures_v18", "sources": sources,
               "figures": [first, second], "created_utc": time.time(), "model_forwards": 0, "fits": 0,
               "new_test_predictions": 0, "new_bootstrap_draws": 0, "scope": "Plots of already stored preflight evidence only"})


def reports(preflight, stop, summary, projection, old_source, old_rows, params_source, params):
    estimate = preflight["estimate"]
    projected, excess = estimate["projected_full_campaign_upper_s"], stop["projected_excess_s"]
    small = next(r for r in read_json(OLD / "preflight_checks.json")["bolt_models"] if r["model"] == "SMALL")
    family_revision = small["id"] + " @ " + small["revision"] + " (literal parent checkpoint)"
    fairness = ["# Matched learned E/D — fairness and execution boundary", "",
                "[확인] 실행 상태는 BLOCKED_STOP_RESOURCE다. 아래는 새 학습 전 실제 loaded enumeration과 parent 계약이다. 새 matched training/inference 비용은 unavailable이며 과거 receipt를 새 cost probe로 표시하지 않는다.", "",
                "```text", "Shared contract", "model_family_revision: " + family_revision,
                "context_horizon: L=512, H=48; FP32; TF32 off", "point_forecast: native inverse-scaled q=0.5, first 48 steps",
                "input: original C channels; B*K independent latent TSFM rows for both arms",
                "preprocessing: literal TRAIN-only normalization, imputation, channel order, origins and observed-target mask",
                "exposure: all three TEST groups were exposed before this follow-up; no independent confirmation",
                "new_fit_count: 0; new_TEST_model_predictions: 0; optimizer_updates: 0", "```", ""]
    for r in summary:
        ds = r["dataset"]
        receipt = preflight["units"][ds]["model_receipt"]
        old = params[ds][0]["receipt"]
        fairness += ["```text", f"{LABELS[ds]}: C={r['C']}, K={r['K']}",
                     "field                         MATCHED_LEARNED_ED(step0)      ORIGINAL_LEVEL(reused)",
                     f"backbone_parameter_count      {receipt['parameter_count'] - receipt['trainable_parameter_count']:<29,d}{old['backbone_parameter_count']:,}",
                     f"total_parameter_count         {receipt['parameter_count']:<29,d}{old['parameter_count']:,}",
                     f"trainable_component_count     {r['ED_trainable_parameters']:<29,d}{old['original_fitted_parameter_count']:,} G (historically fitted)",
                     f"parameter_bytes               {receipt['parameter_bytes']:<29,d}{old['parameter_bytes']:,}",
                     f"buffer_bytes                  {receipt['buffer_bytes']:<29,d}{old['buffer_bytes']:,}",
                     f"total_tensor_bytes            {receipt['total_tensor_bytes']:<29,d}{old['parameter_bytes'] + old['buffer_bytes']:,}",
                     "E/D                           bias=False, trainable          fixed literal PCA",
                     "initialization                E=U.T; D=U                    TRAIN PCA U",
                     "G / level persistence         absent / absent                fitted G / included",
                     "training gradient path        through frozen backbone to E  external residual G only",
                     "new matched cost / resident   unavailable, gate prevented    unavailable, gate prevented",
                     "```", "",
                     f"[확인] ED source: preflight.json `/units/{ds}/model_receipt`; PCA SHA256 `{receipt['basis_values_sha256']}`. LEVEL count source: `{params_source['path']}` `{params[ds][0]['pointer']}`와 두 번째 seed receipt. LEVEL 값은 이전 load 기록이며 이번 회차의 cost 측정이 아니다.", ""]
    fairness += ["[확인] 실제 ED enumeration은 170 / 104 / 252다. LEVEL G 17,920과 parameter matched가 아니다. 전체 배포 가중치에는 동일한 47,718,016 backbone parameters가 포함되므로 작은 adapter count만으로 memory/latency 우위를 말할 수 없다.", "",
                 "[확인] 공식 LinearAutoEncoder는 bias 기본값 True를 사용한다. 본 bias=False, literal PCA 초기화, observed-mask channel-macro forecast MSE만 쓰는 arm은 full AdaPTS reproduction이 아니다. 구조 선택을 좁혀 묻는 control이다. [공식 source pin](https://github.com/abenechehab/AdaPTS/blob/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapters.py#L348-L395), [binding](source_binding.json).", "",
                 f"Source SHA256: preflight.json `{artifact(HERE / 'preflight.json')['sha256']}`; historical LEVEL enumeration `{params_source['sha256']}`."]
    write_text("FAIRNESS_TABLE.md", "\n".join(fairness))
    table = ["dataset              period      LEVEL_MSE       LEVEL_MAE       LEARNED_ED", "-------------------  ----------  --------------  --------------  --------------------------"]
    for r in old_rows:
        table.append(f"{r['dataset']:<19}  {r['period']:<10}  {r['mse']:<14.9f}  {r['mae']:<14.9f}  unavailable, gate prevented")
    budget_table = ["dataset              four_full_fits_s    post_training_s"]
    for ds in DATASETS:
        row = estimate["units"][ds]
        budget_table.append(f"{ds:<19}  {row['four_fits_upper_s']:>16.3f}  {row['post_training_upper_s']:>15.3f}")
    parity_table = ["dataset              VAL_origins   max_abs_error   E/D_parameters   gradients_mb4/2/1"]
    for r in summary:
        parity_table.append(f"{r['dataset']:<19}  {r['canonical_VAL_origins']:>11}   {r['max_absolute_error']:<13.6e}  {r['ED_trainable_parameters']:>14}   finite/nonzero")
    report = f"""# LEVEL vs matched learned E/D — resource stop report

BLOCKED_STOP_RESOURCE. `full_campaign_complete=false`. [확인] 새 neural fit 0, 새 TEST model prediction 0, optimizer update 0이다. fixed E/D + external residual G가 같은 K의 learned E/D보다 필요한지에 대한 질문은 **아직 답하지 못했다**. 사전 점검 통과를 정확도·훈련 비용·추론 비용 우위로 바꾸지 않는다. 비교 전 계약과 실제 parameter enumeration은 [FAIRNESS_TABLE.md](FAIRNESS_TABLE.md)에 있다.

[확인] 전체 계약의 계획 추정치 {projected:.3f}초(3시간 10분 21.5초)가 승인된 cumulative GPU-job wall-time cap 10,800초를 {excess:.3f}초 넘었다. 계약 15절에 따라 학습 전에 STOP_RESOURCE로 종료했다. 실제 preflight GPU job wall time은 {stop['preflight_actual_gpu_job_wall_s']:.6f}초이며 3시간을 실제로 소비한 것이 아니다. failed job, traceback, 부분 readiness 자료는 `ledger.json`, `preflight_attempt01.json`, `preflight.json`, `resource_stop.json`에 보존됐다. resource gate 실패로 `all_checks_passed=false`인 원본을 유지했다.

이 추정은 **12 fits 모두 120 epochs, fit마다 step0 포함 전체 VAL 121회, early stopping 미가정**을 포함한다. dataset당 zero-update forward+backward **3회 중 최댓값**에 1.25를 곱하고, full-VAL 측정, optimizer CPU allowance, checkpoint allowance, 후속 모델 load 40회 allowance 및 TEST/cost/diagnostics 범위를 넣었다. Peacock 원시 F+B seconds는 0.052150800 / 0.053566600 / 0.036708700이다. 계획용 conservative upper projection이지 실제 convergence time, 통계적 upper bound, 또는 3시간 실행 불가능의 증명이 아니다. 선택 checkpoint의 활성화/시간 변화도 측정하지 않았다. 측정 뒤 estimate를 재조정하거나 scientific scope를 줄이지 않았다.

```text
{chr(10).join(budget_table)}
preflight_at_projection_s: {estimate['measured_preflight_gpu_job_s']:.6f}
projected_total_s: {projected:.6f}
adopted_cap_s: 10800
```

[확인] 답한 부분은 정확한 matched 초기 구조가 실행 가능하다는 제한된 질문이다. 각 parent TRAIN PCA bytes/hash, canonical VAL 전체 출력, q=0.5 원래 채널 순서, backbone frozen/eval 및 encoder까지 autograd를 확인했다. microbatch 4/2/1에서 두 weight의 gradient가 finite/nonzero였고 optimizer.step을 하지 않아 E/D와 backbone 상태 hash가 모두 같았다. dataset마다 2 seeds × 120 epochs의 240개 schedule을 literal parent sampler와 대조했다. 이는 학습 완료나 selected model update 증거가 아니다.

```text
{chr(10).join(parity_table)}
tolerance: elementwise abs_error <= 1e-5 + 1e-4 * abs(reference)
same_path_replay_max_absolute: 0 for all three datasets
```

미실행 항목은 selected LR/checkpoint, selected ED TEST-A/B/combined MSE/MAE, signed bias 비교, paired relative MSE/MAE CI, same-session training-path probe, B1 latency/B4 throughput, allocated/reserved/resident memory와 raw cost blocks, selected checkpoint diagnostics다. 모두 **unavailable, gate prevented**다. 없음은 0 error, equal performance, uncertain CI 또는 equivalence를 뜻하지 않는다. Case A/B/C/D 어느 것도 판정하지 않았다. `DATA_DEPENDENT_NO_GENERAL_WINNER` 역시 실제 dataset별 결과나 CI가 있어야 쓰므로 이번 상태의 대용 문구로 사용하지 않는다.

아래 LEVEL 값은 기존 저장된 TEST 결과의 문맥만 제공한다. 새 ED 비교나 재평가가 아니다. 기존 scorer가 각 seed에서 A/B channel SSE와 observed count를 먼저 합쳐 combined macro를 계산한 뒤, 두 selected seed **loss**를 평균한 저장 row를 그대로 복사했다. prediction ensemble이나 dataset 총점은 만들지 않았다.

```text
{chr(10).join(table)}
```

기존 source: `{old_source['path']}`; SHA256 `{old_source['sha256']}`. 정확한 0-based CSV record/physical line와 seed IDs는 `report_summary.json`의 `reused_LEVEL_rows`에 있다. JSON parent 원본 pointer는 `parent_reuse_manifest.json`의 각 `levels[].source_evaluation`, `source_pointer`로 이어진다. 기존 LEVEL의 숫자를 새 learned model 성능으로 해석할 수 없다.

가장 강한 반론 네 가지는 다음과 같다.

1. **핵심 대조가 아직 없다.** 같은 K의 E/D를 forecast loss로 학습하면 G를 대체할 수 있다는 반론은 그대로 유효하다. 초기 COMPRESS parity와 finite gradient는 이에 대한 반증이 아니다.
2. **선행 연구와 완전히 같지 않다.** [ICML 2025 AdaPTS](https://proceedings.mlr.press/v267/benechehab25a.html)와 [공식 pinned supervised path](https://github.com/abenechehab/AdaPTS/blob/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapts.py#L373-L559)를 source audit했다. 공식 LinearAutoEncoder의 bias 기본값은 True, 본 arm은 False다. 공식 default reconstruction coefficient 0.0와 forecast-MSE 경로를 참고했지만 full AdaPTS probabilistic framework를 재현하지 않았다. 공식 `FM.eval()`와 optimizer exclusion만으로 모든 backbone parameter의 requires_grad=False를 추정하지 않았다.
3. **parameter 수와 gradient path가 다르다.** E/D 170/104/252와 historical G 17,920은 parameter matched가 아니다. 같은 압축 손실 처리 전략을 묻는 control이다. backbone-through backprop 비용은 측정할 핵심 tradeoff지만 새 LEVEL/ED 동시 비용 probe가 없으므로 preflight timing/peak memory로 우위를 주장하지 않는다.
4. **노출된 세 dataset과 planning projection의 범위가 좁다.** TEST가 이미 노출된 후속 개발 자료다. 세 자료의 결과가 생겨도 범용성/독립 확인을 뜻하지 않는다. 현재 시간 추정은 짧은 zero-update 측정과 보수적 allowance에 의존하고 실제 early stopping은 더 짧을 수 있다.

판단을 바꿀 증거는 동일 recipe의 최대 12 TRAIN/VAL fits, TEST 이전 joint selection seal, 선택 LR의 6 model TEST predictions, A/B 경계를 지키는 paired bootstrap와 동일 session의 training/inference cost raw blocks다. ED lower MSE CI excludes 0 및 유사/낮은 training cost이면 Case A, ED 더 정확하지만 상당히 높은 cost이면 Case B, LEVEL lower MSE CI excludes 0 및 낮은 training cost이면 Case C, 실제 결과 혼재/CI0이면 Case D를 검토한다. 기존 1.25/2.0 operational cost bins는 사전 해석 기준일 뿐 equivalence margin이 아니며 continuous ratios와 raw ranges를 함께 보아야 한다.

```text
claim_changes:
  keep: 기존 LEVEL 관측 결과의 원래 범위; literal matched step0 구조와 zero-update gradient feasibility
  shrink: matched learned E/D 검증 완료 -> 학습 전 readiness만 확인, resource gate에서 중단
  hold: external residual G 필요성; trained accuracy winner; backprop/inference cost advantage; novelty/generalization
```

최소 다음 계획은 scientific recipe와 3×2LR×2seed=12 fits / selected TEST 6 models를 유지하면서 **resource budget만 명시적으로 재검토**하는 것이다. 예를 들어 14,400초(4시간)는 검토 가능한 상한 예시이며 아직 승인되지 않았다. 원래 STOP_RESOURCE attempt를 보존하고 변경 예산에 대한 continuation gate와 새 source binding을 검토한 뒤에만 재개할 수 있다. 현재 `train_v18.py`는 STOP_RESOURCE preflight를 거부한다. 단순 cap 편집, 자동 fit, bias/K/모델/보조 loss sweep을 실행하지 않는다.

남은 gap은 (1) 승인된 실행 예산, (2) TRAIN/VAL 선택·TEST paired accuracy, (3) matched 비용·선택 checkpoint diagnostics다. 이 회차에서 실행 가능한 부분만 audit하고 핵심 비교는 BLOCKED로 남긴다.

[확인] `verify_v18.py`의 PASS는 저장된 stop 기록과 source/hash/수학의 **자체 검산**에만 해당한다. `full_campaign_complete=false`이며 독립 재현, full AdaPTS reproduction 또는 novelty 인증이 아니다. verifier는 모델 forward/fit/TEST/bootstrap을 실행하지 않고 이미 저장된 step0 VAL score만 재검산한다. 미실행 mandatory checks는 `gate_prevented`로 기록한다. 결과는 [예산 그림](figures/resource_projection.png), [초기화 점검 그림](figures/preflight_readiness.png), source CSV 및 `figures/manifest.json`으로 추적할 수 있다.

```text
BLOCKED
remote_sha: final_checks.json live_remote_main_SHA receipt; publication SHA is recorded after push
new_fits: 0
new_test_predictions: 0
selected_lr: robin/peacock/jena = unavailable, gate prevented
combined_mse: LEVEL = historical rows above; LEARNED_ED = unavailable, gate prevented
paired_relative_mse_ci: unavailable, gate prevented
matched_training_cost: unavailable, gate prevented
inference_cost: unavailable, gate prevented
verdict: unavailable; no scientific Case A/B/C/D
```
"""
    write_text("FINAL_REPORT_KO.md", report)
    write_text("PRESENTATION_PATCH.md", """# Presentation correction — suggestion only

[확인] 이번 회차는 BLOCKED_STOP_RESOURCE이며 기존 PPT/PDF를 수정하지 않았다. 기존 결과 슬라이드에 matched learned E/D 결과가 완료됐다는 표현을 추가하면 안 된다.

발표에 넣는다면 두 새 그림과 다음 문장만 제안한다: “동일 parent PCA로 초기화한 matched learned E/D는 전체 VAL parity와 encoder-through-backbone gradient 점검을 통과했다. 다만 전체 12 fits + 후속 평가의 계획 추정이 3시간 예산을 넘어 학습 전에 중단했다. residual G의 필요성은 아직 판정하지 못했다.”

`figures/resource_projection.png`는 계획용 projection이며 실제 convergence/arm cost 우위 그림이 아니다. `figures/preflight_readiness.png`는 zero-update 점검과 actual parameter enumeration이다. 두 그림 모두 accuracy winner, novelty 또는 독립 확인의 증거로 사용하지 않는다. full AdaPTS reproduction 표현을 쓰지 않고 bias=False 차이와 이미 노출된 세 자료라는 범위를 함께 설명한다.

keep: 기존 LEVEL 결과의 원래 한정 범위와 검증된 step0 readiness.
shrink: “matched learned E/D comparison complete”를 “preflight passed science checks, resource gate stopped execution”로.
hold: G 필요성, trained winner, backprop/inference 비용 주장. 재개 예산 검토 전 자동 후속 실험 없음.
""")
    write_text("README.md", """# LEVEL vs matched learned E/D — v18 resource stop

Execution: **BLOCKED_STOP_RESOURCE**. Full campaign complete: **false**. New fits / TEST model predictions / optimizer updates: **0 / 0 / 0**. Scientific verdict unavailable.

Read [FINAL_REPORT_KO.md](FINAL_REPORT_KO.md) and [FAIRNESS_TABLE.md](FAIRNESS_TABLE.md) first. [PRIOR_ART.md](PRIOR_ART.md) and `source_binding.json` bind official AdaPTS sources and the bias difference. `REQUEST.txt` is the literal adopted contract. Original results, decks, failed attempts and unrelated files are preserved.

![Stored full-scope planning projection and adopted resource cap](figures/resource_projection.png)

![Stored step0 parity and actual parameter enumeration; no trained comparison](figures/preflight_readiness.png)

`preflight.json`, `preflight_attempt01.json`, `preflight_initial_models.json`, `ledger.json`, `resource_stop.json` contain executed zero-update evidence and the resource failure. The planning projection is 11,421.533 s against 10,800 s; actual preflight GPU-job wall time is 21.847596 s. Three timed zero-update backward samples and a 1.25 factor do not measure convergence time or prove 3 h infeasibility. All 12 full 120-epoch fits and 121 full VAL passes per fit were assumed; no early stopping or scope reduction.

`report_v18.py` reads receipts and historical LEVEL CSV rows, writes the four reports, `report_summary.json`, `preflight_summary.csv`, `resource_projection.csv` and two PNG/SVG figures plus exact source/row/hash manifest. It performs no model calls, training, TEST, rescore or bootstrap. `verify_v18.py` audits this stopped attempt, including stored step0 VAL math; a future completed campaign requires a full result verifier.

CPU-only commands from repository root (no new fitting):

```powershell
.venv/Scripts/python.exe research/level_matched_learned_ed_v18_20261006/report_v18.py
.venv/Scripts/python.exe research/level_matched_learned_ed_v18_20261006/verify_v18.py
```

`final_checks.json` PASS means an honest-stop saved-artifact self-check, not full completion, independent reproduction, full AdaPTS reproduction or novelty. Missing selection, ED accuracy, paired CIs, matched costs and selected-checkpoint diagnostics are gate prevented and intentionally absent; no dummy scientific files were created. Static source review is not executed training evidence. The original `train_v18.py` rejects this STOP_RESOURCE preflight.

Minimum next plan: preserve the failed attempt; review a resource-only amendment and continuation gate/source binding, retaining the same 12-fit grid, 6 selected TEST models and scientific recipe. A 4 h cap is an unapproved example. No automatic follow-up experiment.
""")
    save_json(HERE / "report_summary.json", {"schema": "matched_ed_blocked_report_v18", "status": "BLOCKED_STOP_RESOURCE",
              "full_campaign_complete": False, "scientific_verdict": None, "new_fits": 0, "new_test_predictions": 0,
              "optimizer_updates": 0, "projected_full_campaign_upper_s": projected,
              "actual_preflight_gpu_job_wall_s": stop["preflight_actual_gpu_job_wall_s"],
              "selected_lr": {ds: None for ds in DATASETS}, "paired_relative_mse_ci": None,
              "matched_training_cost": None, "inference_cost": None, "unavailable_reason": "Resource gate prevented all TRAIN/VAL fits",
              "reused_LEVEL_source": old_source, "reused_LEVEL_rows": old_rows,
              "historical_LEVEL_parameter_source": params_source, "historical_LEVEL_parameter_pointers": {ds: [r["pointer"] for r in params[ds]] for ds in DATASETS},
              "sources": {name: artifact(HERE / name) for name in ("preflight.json", "resource_stop.json", "source_binding.json", "protocol.json")},
              "model_forwards": 0, "fits": 0, "new_bootstrap_draws": 0, "created_utc": time.time()})


def main():
    preflight, stop = read_json(HERE / "preflight.json"), read_json(HERE / "resource_stop.json")
    ledger = read_json(HERE / "ledger.json")
    if preflight["status"] != "STOP_RESOURCE" or stop["full_campaign_complete"] is not False or ledger["fits"] or ledger["test_predictions"]:
        raise RuntimeError("This generator reports only the saved pre-training STOP_RESOURCE state")
    for receipt in preflight["source_artifacts"]:
        artifact(receipt["path"], receipt["sha256"])
    old_source, old_rows = historical_level_rows()
    params_source, params = level_parameter_receipts()
    summary, projection = prepare_tables(preflight, params)
    reports(preflight, stop, summary, projection, old_source, old_rows, params_source, params)
    figures(summary, projection, preflight, params_source)
    print(json.dumps({"execution_status": "BLOCKED_STOP_RESOURCE", "full_campaign_complete": False,
                      "new_fits": 0, "new_test_predictions": 0, "figures": 2}))


if __name__ == "__main__":
    main()
