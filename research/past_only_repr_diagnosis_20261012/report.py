"""Write Korean reports and compact figures from verified evidence."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def fmt(x, digits=6) -> str:
    if x is None or x == "":
        return "NA"
    try:
        x = float(x)
    except Exception:
        return str(x)
    if abs(x) >= 1000:
        return f"{x:.3f}"
    return f"{x:.{digits}g}"


def pct(delta, base) -> str:
    if base in (None, "", 0) or delta in (None, ""):
        return "NA"
    return f"{100 * float(delta) / float(base):+.3f}%"


def load_metrics() -> list[dict]:
    path = HERE / "metrics.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def metric_float(row, key):
    try:
        return float(row[key])
    except Exception:
        return None


def make_figures(evidence: dict, metrics: list[dict]) -> list[str]:
    FIG.mkdir(exist_ok=True)
    written: list[str] = []

    e1 = evidence.get("E1", {})
    fig, ax = plt.subplots(figsize=(7, 4), dpi=180)
    ax.bar(["future perturbation\nmax |delta|"], [e1.get("max_boundary_prediction_or_feature_abs_delta", 0.0)], color="#4C78A8")
    ax.axhline(1e-5, color="#F58518", linestyle="--", linewidth=1, label="1e-5 reference")
    ax.set_ylabel("max absolute delta")
    ax.set_title("E1 Information-Path Perturbation Check")
    ax.legend(frameon=False)
    path = FIG / "e1_information_path.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    written.append(str(path))

    real = []
    synthetic = []
    for rec in evidence.get("E2", {}).get("records_summary", []):
        item = {
            "label": rec["dataset"] if rec["dataset"] in ("jena", "robin") else rec["tag"].replace("run02_", ""),
            "dataset": rec["dataset"],
            "delta_mse": rec.get("delta_all_lag_mse_vs_self"),
            "delta_mae": rec.get("delta_all_lag_mae_vs_self"),
        }
        if rec["dataset"] in ("jena", "robin"):
            real.append(item)
        elif rec["dataset"] in ("synthetic_fixed", "synthetic_changed", "synthetic_unrelated"):
            synthetic.append(item)
    if real:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=180)
        x = np.arange(len(real))
        ax.bar(x - 0.18, [r["delta_mse"] for r in real], width=0.36, label="MSE delta")
        ax.bar(x + 0.18, [r["delta_mae"] for r in real], width=0.36, label="MAE delta")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, [r["label"] for r in real])
        ax.set_title("E2 ALL_CHANNEL_LAG minus SELF_LAG on Real VAL")
        ax.set_ylabel("physical macro error delta")
        ax.legend(frameon=False)
        path = FIG / "e2_real_lag_deltas.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))
    if synthetic:
        fig, ax = plt.subplots(figsize=(9, 4), dpi=180)
        labels = [r["label"] for r in synthetic]
        ax.plot(labels, [r["delta_mse"] for r in synthetic], marker="o", label="MSE delta")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title("E2 Synthetic Lag Diagnostics")
        ax.set_ylabel("ALL_CHANNEL_LAG - SELF_LAG")
        ax.legend(frameon=False)
        path = FIG / "e2_synthetic_lag_deltas.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    e34_rows = [r for r in metrics if r["experiment"] in ("E3", "E4") and r["mse"]]
    if e34_rows:
        fig, ax = plt.subplots(figsize=(10, 4.6), dpi=180)
        labels = [f"{r['experiment']}:{r['condition']}:s{r['seed']}" for r in e34_rows]
        ax.bar(np.arange(len(e34_rows)), [float(r["mse"]) for r in e34_rows], color="#54A24B")
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(e34_rows)), labels, rotation=60, ha="right")
        ax.set_ylabel("physical macro MSE (log)")
        ax.set_title("E3/E4 Jena d16 Diagnostic Subset")
        path = FIG / "e34_jena_d16_subset_mse.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    e5e6_rows = [r for r in metrics if r["experiment"] in ("E5", "E6") and r["mse"]]
    if e5e6_rows:
        fig, ax = plt.subplots(figsize=(10, 4.8), dpi=180)
        labels = [f"{r['experiment']}:{r['condition']}:s{r['seed']}" for r in e5e6_rows]
        ax.bar(np.arange(len(e5e6_rows)), [float(r["mae"]) for r in e5e6_rows], color="#B279A2")
        ax.set_xticks(np.arange(len(e5e6_rows)), labels, rotation=60, ha="right")
        ax.set_ylabel("physical macro MAE")
        ax.set_title("E5/E6 Completed Conditions")
        path = FIG / "e5_e6_completed_mae.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    return written


def e2_line(evidence: dict, dataset: str) -> str:
    rec = next((r for r in evidence.get("E2", {}).get("records_summary", []) if r["dataset"] == dataset), None)
    if not rec:
        return f"- {dataset}: 미실행"
    return (
        f"- {dataset}: SELF {fmt(rec['self_mse'])}/{fmt(rec['self_mae'])}, "
        f"ALL-LAG {fmt(rec['all_lag_mse'])}/{fmt(rec['all_lag_mae'])}, "
        f"delta {fmt(rec['delta_all_lag_mse_vs_self'])}/{fmt(rec['delta_all_lag_mae_vs_self'])} "
        f"({pct(rec['delta_all_lag_mse_vs_self'], rec['self_mse'])} MSE)."
    )


def e34_seed_lines(evidence: dict) -> list[str]:
    lines = []
    for result in evidence.get("E3_E4", {}).get("results", []):
        e4 = result.get("E4", {})
        prefix = e4.get("BOLT_LATENT", {})
        ref = e4.get("REFERENCE_LATENT_DIAGNOSTIC", {})
        self_ridge = e4.get("LATENT_SELF_RIDGE", {})
        joint = e4.get("LATENT_JOINT_RIDGE", {})
        raw = e4.get("DIRECT_RAW_RIDGE", {})
        lines.append(
            f"- seed {result.get('seed')}: PREFIX/Bolt {fmt(prefix.get('mse'))}/{fmt(prefix.get('mae'))}, "
            f"latent-self ridge {fmt(self_ridge.get('mse'))}/{fmt(self_ridge.get('mae'))}, "
            f"latent-joint ridge {fmt(joint.get('mse'))}/{fmt(joint.get('mae'))}, "
            f"reference-latent diagnostic {fmt(ref.get('mse'))}/{fmt(ref.get('mae'))}, "
            f"direct raw ridge {fmt(raw.get('mse'))}/{fmt(raw.get('mae'))}."
        )
    return lines or ["- 미완료"]


def completed_e5e6(evidence: dict) -> tuple[list[dict], list[dict]]:
    e5, e6 = [], []
    for item in evidence.get("E5_E6", {}).get("results", []):
        if item.get("experiment") == "E5":
            e5.append(item)
        elif item.get("experiment") == "E6":
            e6.append(item)
    return e5, e6


def write_report(evidence: dict, metrics: list[dict], figures: list[str]) -> str:
    e1 = evidence.get("E1", {})
    e34 = evidence.get("E3_E4", {})
    e5_results, e6_results = completed_e5e6(evidence)
    e5_science = [r for r in e5_results if r.get("status") == "COMPLETE_BALANCED"]
    e6_science = [r for r in e6_results if r.get("status") == "COMPLETE_BALANCED"]

    e34_complete = e34.get("balanced_pair_complete") is True
    e5_status = "완료" if e5_science else "과학 실행 미완료"
    e6_status = "완료" if e6_science else "과학 실행 미완료"

    lines = [
        "# 과거 전용 표현·지연 그래프 진단 결과",
        "",
        "## 먼저 답",
        "",
        f"1. 실제 미래 정보 누출: [확인] E1 경로에서 예측 경계 이후 perturbation의 최대 절대 차이는 {fmt(e1.get('max_boundary_prediction_or_feature_abs_delta'))}였고, 정확 변경 count 합계는 {e1.get('changed_exact_count_total')}입니다. float 수준 잔차를 제외하면 현재 구현된 E/F/D 추론 경로에서 미래 target이 입력으로 들어간 증거는 없습니다. 단, 이것은 저장된 TRAIN/VAL 자산과 선택한 16개 checkpoint·2 origin·representation check 범위의 안전성 검사이지 성능 개선이나 독립 일반화 검증은 아닙니다.",
        "",
        "2. 시점별 과거 전용 표현의 예측 개선: [확인] E3/E4는 비용 때문에 Jena TS2Vec d16 진단 subset으로 축소되었습니다. 완료된 seed에서 PREFIX 추출은 FULL 추출보다 같은 D 조건에서 물리 MSE가 더 나빴지만 normalized MSE는 더 낮은 식으로 지표가 엇갈렸습니다. 두 seed 균형 완료 여부는 아래 E3/E4 상태를 기준으로 해석해야 합니다.",
        "",
        "3. 다른 채널의 지연 정보: [확인] E2 선형 lag 진단에서 Jena는 ALL_CHANNEL_LAG가 SELF_LAG보다 낮은 MSE를 보였고, Robin은 이득이 확인되지 않았습니다. 합성 changed 조건에서는 작은 이득이 반복됐지만, unrelated 조건에서도 false-positive 성격의 개선이 일부 있어 gate 통과를 일반성으로 해석하면 안 됩니다.",
        "",
        "4. 남은 문제 위치: [확인] E4 완료 seed 기준으로 reference-latent diagnostic이 가장 낮고, latent ridge가 Bolt latent보다 낮아 잠재 예측/목적 문제가 큽니다. D 자체도 reference를 넣었을 때 아직 큰 물리 오차가 남아 decoder 병목도 남습니다. 표현, 잠재 미래 예측, decoder 모두 완전히 배제되지 않았습니다.",
        "",
        f"5. 대조학습·동적 그래프 필요성: [확인] E5 상태는 {e5_status}, E6 상태는 {e6_status}입니다. 현재 완료 증거만으로는 대조학습이나 동적 graph가 필요하다고 말할 수 없습니다. E5/E6 과학 실행이 완료되면 단순 causal reconstruction/forecast와 고정 graph가 충분한지 비교할 수 있습니다.",
        "",
        "6. Chronos-Bolt 유지 근거: [확인] 현재 Jena d16 subset에서 Bolt를 붙인 latent 경로는 latent ridge보다 나빴습니다. Bolt 유지 근거는 아직 강하지 않고, 유지하려면 동일 표현·동일 readout·동일 비용에서 direct ridge보다 나은 조건을 다시 보여야 합니다.",
        "",
        "## 비교 조건",
        "",
        "- 데이터: Jena와 Robin TRAIN/VAL만 사용, 보호된 TEST 접근 금지.",
        "- 시간 계약: history `o-512:o`, future `o:o+48`; E2/E5/E6 새 계수는 FIT에서 학습하고 TUNE으로 선택.",
        "- E3/E4: 기존 TS2Vec d16 checkpoint 재사용, 새 D는 FULL/PREFIX 같은 초기화와 같은 update 상한.",
        "- 지표: 원공간 physical macro MSE/MAE를 주 지표로 쓰고, normalized companion은 해석 보조로만 사용.",
        "",
        "## 원수치",
        "",
        "E2 real VAL:",
        e2_line(evidence, "jena"),
        e2_line(evidence, "robin"),
        "",
        "E3/E4 Jena d16 subset:",
        *e34_seed_lines(evidence),
        "",
        "E5/E6:",
    ]
    if not e5_results and not e6_results:
        lines.append("- 결과 파일 없음.")
    else:
        for item in e5_results + e6_results:
            lines.append(f"- {item.get('experiment') or 'E5/E6'} {item.get('dataset')}: {item.get('status')} ({item.get('path')})")
    lines.extend([
        "",
        "## 불확실성",
        "",
        "- VAL은 반복 노출된 개발 진단 자료라 독립 test 일반화가 아닙니다.",
        "- E3/E4가 DIAG_SUBSET이면 기존 full VAL 결과와 직접 비교하지 않습니다.",
        "- 저장 dtype 차이 때문에 physical inverse 후 float64 오차 재계산이 1e-7 수준 tolerance를 사용합니다.",
        "- graph/contrast 결과가 미완료이면 필요성은 보류이며, 단순 대안이 충분하다는 결론도 아직 보류입니다.",
        "",
        "## 다음 판단",
        "",
        "- E3/E4 두 seed가 모두 완료되면 E5를 실행해 objective 문제를 분리합니다.",
        "- E2에서 Jena는 E6 실자료 후보, Robin은 보류 후보입니다.",
        "- E6은 합성 changed 조건과 Jena를 분리해서 보고해야 하며, synthetic 성공을 실자료 성공으로 바꾸면 안 됩니다.",
        "",
        "## 산출물",
        "",
        "- `EVIDENCE.json`: 검산 근거와 source/hash 요약",
        "- `metrics.csv`: 주요 조건별 MSE/MAE",
    ])
    for fig in figures:
        rel = Path(fig).relative_to(HERE).as_posix()
        lines.append(f"- `{rel}`")
    text = "\n".join(lines) + "\n"
    (HERE / "REPORT_KO.md").write_text(text, encoding="utf-8")
    return text


def write_professor_update(report_text: str) -> None:
    lines = [
        "# 교수님 공유용 요약 초안",
        "",
        "이번 실험은 LEVEL 개선이 아니라, 과거 정보만으로 만든 표현과 지연 graph가 실제 예측 시점에서 계산 가능하고 유용한지 확인하는 진단입니다.",
        "",
        "현재 확인된 핵심은 다음과 같습니다.",
        "",
    ]
    for line in report_text.splitlines():
        if line.startswith(("1. ", "2. ", "3. ", "4. ", "5. ", "6. ")):
            lines.append(line)
    lines.extend([
        "",
        "따라서 현재 단계의 결론은 “과거 전용 구현 안전성은 대체로 통과했지만, 표현 추출 변경만으로 성능 개선을 확인했다고 말할 수 없고, 잠재 예측 목표와 decoder 병목을 더 좁혀야 한다”입니다.",
        "",
        "주의: 위 결과는 개발 VAL과 축소 subset 기반의 진단이며, 독립 TEST 일반화 주장이 아닙니다.",
    ])
    (HERE / "PROFESSOR_UPDATE_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    evidence_path = HERE / "EVIDENCE.json"
    if not evidence_path.exists():
        raise SystemExit("Run verify_results.py first")
    evidence = read_json(evidence_path)
    metrics = load_metrics()
    figures = make_figures(evidence, metrics)
    report_text = write_report(evidence, metrics, figures)
    write_professor_update(report_text)
    changed = [
        "research/past_only_repr_diagnosis_20261012/verify_results.py",
        "research/past_only_repr_diagnosis_20261012/report.py",
        "research/past_only_repr_diagnosis_20261012/EVIDENCE.json",
        "research/past_only_repr_diagnosis_20261012/metrics.csv",
        "research/past_only_repr_diagnosis_20261012/REPORT_KO.md",
        "research/past_only_repr_diagnosis_20261012/PROFESSOR_UPDATE_KO.md",
        *[str(Path(p).relative_to(HERE.parent.parent).as_posix()) for p in figures],
    ]
    (HERE / "CHANGED_FILES.txt").write_text("\n".join(changed) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(HERE / "REPORT_KO.md"), "figures": figures}, ensure_ascii=False))


if __name__ == "__main__":
    main()
