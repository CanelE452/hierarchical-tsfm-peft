"""Write analysis tables, figures, and Korean report for the Bi-Hash pilot."""
from pathlib import Path
import csv
import json
import math
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "report_v1"
FIG = OUT / "figures"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def pct(value):
    return f"{value:+.2f}%"


def fmt(value, nd=6):
    if value is None:
        return "NA"
    return f"{value:.{nd}f}"


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def summarize(dev, retrieval):
    base = {}
    dev_rows = []
    for case in dev["cases"]:
        if case["method"] == "BASE":
            base[(case["dataset"], case["seed"])] = case["selected_checkpoint"]["forecast"]["tune"]["normalized"]["mse"]
    for case in dev["cases"]:
        b = base[(case["dataset"], case["seed"])]
        best = case["selected_checkpoint"]
        dev_rows.append(
            {
                "dataset": case["dataset"],
                "seed": case["seed"],
                "method": case["method"],
                "path": "BOLT",
                "selected_update": case["selected_update"],
                "mse": best["forecast"]["tune"]["normalized"]["mse"],
                "mae": best["forecast"]["tune"]["normalized"]["mae"],
                "delta_pct": 100 * (best["forecast"]["tune"]["normalized"]["mse"] / b - 1),
                "hash_top1": case.get("diagnostics", {}).get("fit", {}).get("fit_hash_top1_self_retrieval"),
                "latent_before": case.get("diagnostics", {}).get("fit", {}).get("before_q_latent_future_normalized_mse"),
                "latent_after": case.get("diagnostics", {}).get("fit", {}).get("after_q_latent_future_normalized_mse"),
            }
        )
        q = case.get("selected_q_checkpoint")
        if q is not None:
            dev_rows.append(
                {
                    "dataset": case["dataset"],
                    "seed": case["seed"],
                    "method": case["method"],
                    "path": "BOLT_Q",
                    "selected_update": case["selected_q_update"],
                    "mse": q["forecast_q"]["tune"]["normalized"]["mse"],
                    "mae": q["forecast_q"]["tune"]["normalized"]["mae"],
                    "delta_pct": 100 * (q["forecast_q"]["tune"]["normalized"]["mse"] / b - 1),
                    "hash_top1": case.get("diagnostics", {}).get("fit", {}).get("fit_hash_top1_self_retrieval"),
                    "latent_before": case.get("diagnostics", {}).get("fit", {}).get("before_q_latent_future_normalized_mse"),
                    "latent_after": case.get("diagnostics", {}).get("fit", {}).get("after_q_latent_future_normalized_mse"),
                }
            )

    retrieval_rows = []
    for case in retrieval["cases"]:
        b = base[(case["dataset"], case["seed"])]
        for record in case["retrieval"]:
            for alpha, payload in record["candidate_scores"].items():
                score = payload["scores"]["normalized"]
                retrieval_rows.append(
                    {
                        "dataset": case["dataset"],
                        "seed": case["seed"],
                        "method": case["method"],
                        "mode": record["mode"],
                        "alpha_q": float(alpha),
                        "mse": score["mse"],
                        "mae": score["mae"],
                        "delta_pct": 100 * (score["mse"] / b - 1),
                        "entropy": record["retrieval_entropy"],
                        "mean_top1_weight": record["mean_top1_weight"],
                        "causal_bank_min": record["causal_bank_min"],
                        "causal_bank_max": record["causal_bank_max"],
                        "nearest_fit_target_end_lag_min": record["nearest_fit_target_end_lag_min"],
                    }
                )
    best_retrieval = []
    keys = sorted({(r["dataset"], r["seed"], r["method"], r["mode"]) for r in retrieval_rows})
    for key in keys:
        group = [r for r in retrieval_rows if (r["dataset"], r["seed"], r["method"], r["mode"]) == key]
        best_retrieval.append(min(group, key=lambda r: r["mse"]))

    def aggregate(rows, keys):
        out = []
        groups = sorted({tuple(r[k] for k in keys) for r in rows})
        for group_key in groups:
            group = [r for r in rows if tuple(r[k] for k in keys) == group_key]
            values = [r["delta_pct"] for r in group]
            out.append(
                {
                    **{k: v for k, v in zip(keys, group_key)},
                    "n": len(values),
                    "mean_delta_pct": float(np.mean(values)),
                    "min_delta_pct": float(np.min(values)),
                    "max_delta_pct": float(np.max(values)),
                }
            )
        return out

    dev_summary = aggregate(dev_rows, ["path", "method"])
    retrieval_summary = aggregate(best_retrieval, ["method", "mode"])
    return base, dev_rows, dev_summary, retrieval_rows, best_retrieval, retrieval_summary


def make_figures(dev_rows, dev_summary, best_retrieval, retrieval_summary):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

    paired = [r for r in dev_rows if r["method"] != "BASE"]
    labels = [f"{r['dataset']}\\n{r['seed']}\\n{r['method']}\\n{r['path']}" for r in paired]
    values = [r["delta_pct"] for r in paired]
    colors = ["#4C78A8" if r["method"] == "BIHASH" else "#F58518" if r["method"] == "SUMMARY_MSE" else "#B279A2" for r in paired]
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.set_ylabel("Tune normalized MSE delta vs BASE (%)")
    ax.set_title("Paired future auxiliary: improvement is small and seed dependent")
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=65, ha="right")
    fig.savefig(FIG / "paired_aux_delta.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    methods = ["SUMMARY_MSE", "BIHASH", "BIHASH_SHUFFLED"]
    paths = ["BOLT", "BOLT_Q"]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    x = np.arange(len(methods))
    width = 0.35
    for i, path in enumerate(paths):
        ys = []
        for method in methods:
            row = next(r for r in dev_summary if r["path"] == path and r["method"] == method)
            ys.append(row["mean_delta_pct"])
        ax.bar(x + (i - 0.5) * width, ys, width, label=path)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Mean delta vs BASE (%) across 4 dataset-seed pairs")
    ax.set_title("BiHash is not consistently better than its shuffled control")
    ax.legend()
    fig.savefig(FIG / "paired_aux_mean_delta.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    labels = []
    ys = []
    colors = []
    for row in retrieval_summary:
        labels.append(f"{row['method']}\\n{row['mode']}")
        ys.append(row["mean_delta_pct"])
        colors.append("#54A24B" if row["mode"] == "past_continuous" else "#4C78A8" if row["mode"] == "continuous" else "#E45756")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.bar(np.arange(len(ys)), ys, color=colors)
    ax.set_ylabel("Mean best delta vs BASE (%)")
    ax.set_title("Causal memory retrieval: continuous similarity beats hash lookup")
    ax.set_xticks(np.arange(len(ys)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.savefig(FIG / "retrieval_mean_delta.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    xs = [r["mean_top1_weight"] for r in best_retrieval]
    ys = [r["delta_pct"] for r in best_retrieval]
    colors = ["#E45756" if r["mode"] == "bihash" else "#4C78A8" if r["mode"] == "continuous" else "#54A24B" for r in best_retrieval]
    ax.axhline(0, color="black", linewidth=0.8)
    ax.scatter(xs, ys, c=colors, s=55)
    ax.set_xlabel("Mean top-1 retrieval weight")
    ax.set_ylabel("Best delta vs BASE (%)")
    ax.set_title("Hash lookup stays diffuse; strong retrieval comes from continuous distance")
    fig.savefig(FIG / "retrieval_weight_vs_delta.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_report(dev, retrieval, dev_rows, dev_summary, best_retrieval, retrieval_summary):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    fixture = load(HERE / "run_fixture_v2" / "FIXTURE.json")
    resource_files = sorted((HERE / "resources").glob("*.json"))
    resources = [load(p) for p in resource_files]
    total_wall = sum(float(r.get("wall_seconds", 0)) for r in resources if r.get("status") in ("COMPLETE", "FAILED_PRESERVED"))

    def section_rows(rows, fields):
        lines = ["```text", " | ".join(fields)]
        lines.append("-+-".join("-" * len(f) for f in fields))
        for row in rows:
            vals = []
            for field in fields:
                value = row.get(field)
                if isinstance(value, float):
                    value = fmt(value, 6)
                vals.append(str(value))
            lines.append(" | ".join(vals))
        lines.append("```")
        return "\n".join(lines)

    key_dev = [
        {
            "dataset": r["dataset"],
            "seed": r["seed"],
            "method": r["method"],
            "path": r["path"],
            "mse": fmt(r["mse"], 6),
            "delta": pct(r["delta_pct"]),
            "hash_top1": fmt(r.get("hash_top1"), 4),
        }
        for r in dev_rows
        if r["method"] in ("BASE", "BIHASH", "BIHASH_SHUFFLED") and r["path"] in ("BOLT", "BOLT_Q")
    ]
    retr_key = [
        {
            "dataset": r["dataset"],
            "seed": r["seed"],
            "method": r["method"],
            "mode": r["mode"],
            "alpha": r["alpha_q"],
            "mse": fmt(r["mse"], 6),
            "delta": pct(r["delta_pct"]),
            "top1_weight": fmt(r["mean_top1_weight"], 4),
            "min_end_lag": r["nearest_fit_target_end_lag_min"],
        }
        for r in best_retrieval
        if r["method"] == "BIHASH"
    ]
    report = f"""# Bi-Hash 미래 표현 정렬 파일럿 결과

## 결론

[확인] 이 실험에서 ExoTimer식 학습 가능한 hash projection은 실제로 학습되고 encoder/decoder에 gradient를 전달했다. fixture에서 hash 모듈 gradient 합은 {fixture['hash_parameters_receive_gradient_sum_abs']:.6f}, codec gradient 합은 {fixture['codec_parameters_receive_gradient_sum_abs']:.6f}였다.

[확인] 하지만 현재 설계로는 “미래 예측에 도움이 되는 표현을 안정적으로 만들 수 있다”는 결론까지 가지 못한다. 개발용 Jena/Robin 2자료 x 2seed에서 paired BiHash의 일반 BOLT 경로 평균 delta는 {next(r for r in dev_summary if r['path']=='BOLT' and r['method']=='BIHASH')['mean_delta_pct']:.3f}%이고, shuffled BiHash는 {next(r for r in dev_summary if r['path']=='BOLT' and r['method']=='BIHASH_SHUFFLED')['mean_delta_pct']:.3f}%였다. 평균만 보면 shuffled가 더 좋아 보이는 형태까지 나온다.

[확인] `BOLT_Q` 경로에서는 BiHash 평균 delta가 {next(r for r in dev_summary if r['path']=='BOLT_Q' and r['method']=='BIHASH')['mean_delta_pct']:.3f}%로 거의 0에 가깝지만, Robin seed92601만 -1.138% 개선이고 Robin seed92602는 +0.226% 악화다. seed 하나의 양성 신호는 있으나 반복되지 않았다.

[확인] 추론 때 실제로 쓸 수 있는 causal FIT memory retrieval에서는 hash lookup 자체가 약했다. BiHash mode의 top-1 retrieval weight는 대체로 0.02-0.03 수준이고 entropy는 약 5에 가까워 거의 퍼진 조회에 가깝다. 오히려 continuous similarity나 past-continuous similarity가 더 낫다. 이는 ExoTimer의 “학습 가능한 hash 경계”가 이 과거-미래 표현 문제에서 필요한 핵심 요소라고 보기 어렵다는 근거다.

[판단] 가능한가/불가능한가로 자르면, “구현 가능하고 일부 seed에서 작은 개선은 가능”하지만 “논문 기여로 밀 만큼 안정적인 방법은 현재 증거로는 불가능에 가깝다.” 더 유망한 방향은 BiHash 자체보다 causal memory retrieval 또는 continuous past-future similarity를 단순 baseline으로 먼저 세우는 것이다.

## 실험 계약

- 입력은 과거 512 step, 예측은 미래 48 step이다.
- 주지표는 TUNE normalized observed channel-macro MSE다.
- 보호 TEST는 접근하지 않았다. fixture와 실행 결과에 보호 경로 가드가 기록돼 있다.
- 개발용 Jena/Robin만 사용했다. 이전에 여러 번 본 자료이므로 독립 일반화 검증으로 부르지 않는다.
- 학습은 200 update, checkpoint 0/50/100/200, batch size 4, seed 92601/92602다.
- BASE와 동일한 `ANCHOR_NONLINEAR_D` 예측 경로 위에 보조 목적만 바꿨다.
- BiHash는 ExoTimer의 원래 LLM-text/time-series 모달리티 정렬 재현이 아니라, 과거 latent delta와 FIT 미래 latent summary 사이의 전이 실험이다.

## paired auxiliary 결과

![paired auxiliary per case](figures/paired_aux_delta.png)

![paired auxiliary mean](figures/paired_aux_mean_delta.png)

{section_rows(key_dev, ['dataset','seed','method','path','mse','delta','hash_top1'])}

해석은 조심해야 한다. BiHash의 FIT hash self-retrieval top1은 0.0039-0.0195 수준이다. 256개 FIT origin에서 무작위 top1이 약 0.0039라는 점을 고려하면, Robin에서는 신호가 조금 있지만 강한 분리라고 보기 어렵고 Jena에서는 사실상 약하다. 또한 latent future MSE가 줄어드는 경우가 반드시 forecast MSE 개선으로 이어지지 않았다.

## causal memory retrieval 결과

paired auxiliary는 학습 시 실제 future target과 짝을 맞춘다. 그러나 추론 시 실제 future target은 없다. 그래서 추가 학습 없이, FIT future pattern bank 중 `fit_origin + H <= tune_origin`인 항목만 조회하는 causal memory retrieval을 계산했다.

![retrieval mean](figures/retrieval_mean_delta.png)

![retrieval weight](figures/retrieval_weight_vs_delta.png)

{section_rows(retr_key, ['dataset','seed','method','mode','alpha','mse','delta','top1_weight','min_end_lag'])}

[확인] causal bank는 모든 TUNE query에서 최소 256개 항목을 갖고, nearest item도 query origin 이전에 미래 target이 끝난 항목만 사용했다. 따라서 retrieval 값은 미래 입력 누출 없이 계산됐다.

[판단] hash retrieval은 lookup이 너무 퍼진다. BiHash mode는 Jena에서 BASE보다 낫지 않고, Robin의 양성 seed에서는 사실상 `alpha_q=1.0`이 선택되어 retrieval이 아니라 q path 자체가 결과를 만든다. 반면 Jena에서는 continuous/past-continuous retrieval이 BASE보다 최대 1.36% 낮아진다. 이건 “hash 경계”보다 “과거 표현과 FIT future pattern의 연속 거리”가 더 직접적인 후보라는 뜻이다.

## 무엇을 해결했고 무엇이 남았나

[확인] 해결한 것:

```text
1. 기존 KMeans 고정 regime 변형이 원 BiHash가 아니라는 문제를 분리했다.
2. 학습 가능한 hash projection, Gumbel-softmax relaxation, cross-batch retrieval loss, continuous hybrid head를 구현했다.
3. paired future auxiliary와 shuffled future auxiliary를 같은 조건에서 비교했다.
4. 추론 때 쓸 수 없는 future target 문제를 causal FIT memory retrieval로 따로 검사했다.
5. hash가 실제로 분리되는지 top1 retrieval, entropy, top1 weight를 기록했다.
```

[판단] 남은 문제:

```text
1. BiHash의 이진 hash lookup이 충분히 날카롭지 않다.
2. q로 latent future 좌표를 맞추는 학습이 forecast 개선과 안정적으로 연결되지 않는다.
3. 좋은 결과가 나오는 경우도 shuffled 또는 continuous retrieval로 설명 가능한 부분이 있다.
4. 현재 결과는 반복 사용한 개발자료 TUNE이므로 독립 검증이 아니다.
```

## 다음 판단

[판단] 지금 단계에서 새 independent EVAL을 여는 것은 이르다. 개발 단계에서 BiHash가 shuffled control을 안정적으로 이기지 못했고, 추론 가능한 hash retrieval도 약했다. 다음 실험을 한다면 BiHash 이름으로 밀기보다 다음 두 가지를 먼저 비교해야 한다.

```text
A. causal continuous memory retrieval:
   hash 없이 과거 표현과 FIT future pattern을 연속거리로 조회한다.
   Jena에서 신호가 더 컸으므로 단순 기준군으로 세워야 한다.

B. retrieval-aware training:
   paired future target loss가 아니라, query origin 이전 bank만 조회하는 구조를 학습 중에도 사용한다.
   이때도 shuffled bank, time-shifted bank, alpha=1 q-only 대조가 필요하다.
```

## 산출물

- 원결과: `run_dev_v1/RESULT.json`, `run_retrieval_v2/RESULT.json`
- 정리 CSV: `tables/paired_aux_rows.csv`, `tables/paired_aux_summary.csv`, `tables/retrieval_best_rows.csv`, `tables/retrieval_summary.csv`
- 그림: `figures/paired_aux_delta.png`, `figures/paired_aux_mean_delta.png`, `figures/retrieval_mean_delta.png`, `figures/retrieval_weight_vs_delta.png`
- 코드: `bihash_pilot.py`, `retrieval_posthoc.py`, `make_report.py`

## 재현 정보

```text
HEAD: {head}
dev status: {dev['status']}
retrieval status: {retrieval['status']}
resource receipt files: {len(resource_files)}
total recorded wall seconds including failed smoke attempts: {total_wall:.2f}
```
"""
    (OUT / "REPORT_KO.md").write_text(report, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = load(HERE / "run_dev_v1" / "RESULT.json")
    retrieval = load(HERE / "run_retrieval_v2" / "RESULT.json")
    base, dev_rows, dev_summary, retrieval_rows, best_retrieval, retrieval_summary = summarize(dev, retrieval)
    write_csv(
        OUT / "tables" / "paired_aux_rows.csv",
        dev_rows,
        ["dataset", "seed", "method", "path", "selected_update", "mse", "mae", "delta_pct", "hash_top1", "latent_before", "latent_after"],
    )
    write_csv(OUT / "tables" / "paired_aux_summary.csv", dev_summary, ["path", "method", "n", "mean_delta_pct", "min_delta_pct", "max_delta_pct"])
    write_csv(
        OUT / "tables" / "retrieval_rows.csv",
        retrieval_rows,
        ["dataset", "seed", "method", "mode", "alpha_q", "mse", "mae", "delta_pct", "entropy", "mean_top1_weight", "causal_bank_min", "causal_bank_max", "nearest_fit_target_end_lag_min"],
    )
    write_csv(
        OUT / "tables" / "retrieval_best_rows.csv",
        best_retrieval,
        ["dataset", "seed", "method", "mode", "alpha_q", "mse", "mae", "delta_pct", "entropy", "mean_top1_weight", "causal_bank_min", "causal_bank_max", "nearest_fit_target_end_lag_min"],
    )
    write_csv(OUT / "tables" / "retrieval_summary.csv", retrieval_summary, ["method", "mode", "n", "mean_delta_pct", "min_delta_pct", "max_delta_pct"])
    make_figures(dev_rows, dev_summary, best_retrieval, retrieval_summary)
    make_report(dev, retrieval, dev_rows, dev_summary, best_retrieval, retrieval_summary)
    summary = {
        "status": "COMPLETE",
        "report": str((OUT / "REPORT_KO.md").relative_to(ROOT)),
        "figures": [str(p.relative_to(ROOT)) for p in sorted(FIG.glob("*.png"))],
        "tables": [str(p.relative_to(ROOT)) for p in sorted((OUT / "tables").glob("*.csv"))],
    }
    bp = OUT / "SUMMARY.json"
    bp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
