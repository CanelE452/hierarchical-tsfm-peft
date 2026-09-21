import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .audit import ROOT, RESULTS, sha256, utc_now, write_json


def finalize():
    decision = json.loads((RESULTS / "CONTINUATION_DECISION.json").read_text())
    stage_b = RESULTS / "tourism_confirmation"
    has_b = (stage_b / "PROGRESS.json").exists() and json.loads((stage_b / "PROGRESS.json").read_text())["status"] == "COMPLETE"
    if has_b:
        b = pd.read_csv(stage_b / "SEED_EFFECTS.csv")
        controls = b[b.comparator.isin(["LORA_SELF", "LORA_POOL"])]
        favorable = ((controls.primary_difference < -1e-8).all() and (controls.bottom_relative_difference <= 0.05).all())
        label = "PROCEED_HIER_PEFT_METHOD_RESEARCH" if favorable else "WEAK_SIGNAL_HOLD"
    else:
        assert decision["decision"] == "STOP_CURRENT_HIER_ADAPTER_AFTER_LABOUR_SCREEN", "Stage B required by passed project gate"
        label = "STOP_CURRENT_HIER_ADAPTER"
    lines = ["# Hierarchical TSFM PEFT 실행 보고", "", f"최종 판단: **{label}**", "",
             "[확인] MASTER 실행 계약에 따라 Labour screen을 완료했습니다. 이 판단은 후속 연구 예산에 관한 결정이며 논문 PASS 선언이 아닙니다.", "",
             f"Labour continuation: `{decision['decision']}`", "", "## 환경과 검산", "",
             "Python 3.11.16, PyTorch 2.10.0+cu128 / CUDA 12.8, RTX 4070 12GB. 정확한 전체 버전·wheel provenance는 ENVIRONMENT.json과 requirements-lock.txt에 기록했습니다.",
             "Chronos model revision은 MODEL_SOURCE_MANIFEST.json에 고정했습니다. F0 공식 parity와 세 seed의 adapter step0 parity는 실제 Labour 입력에서 차이 0으로 확인했습니다.",
             "PEFT의 embedding accessor 호환성 오류는 기존 shared embedding을 반환하도록 연결하여 해결했고, 파라미터 수·동결 가중치·예측 parity를 검증했습니다.", "",
             "## 데이터와 평가 범위", "",
             "공식 datasetsforecast.HierarchicalData로 Labour와 TourismLarge를 받았으며 전체 시간축의 합계·누락·중복을 감사했습니다. 원본과 weights는 Git에서 제외했습니다.",
             "Context 48개월, horizon 12개월입니다. TRAIN/CALIBRATION/VALIDATION/TEST를 분리했고 TEST는 LR·checkpoint·reconciler 선택에 쓰지 않았습니다.",
             "Primary는 series RMSSE → 각 공식 level 평균 → level 동일 가중 평균입니다. denominator는 TRAIN seasonal scale입니다.",
             "MinT residual은 13개 CALIBRATION origins × 12 horizons이며, 서로 겹치는 달의 예측 오차도 별도 관측으로 유지했습니다. TEST의 37 rolling origins 또한 겹치므로 독립 표본처럼 통계적 유의성을 주장하지 않습니다.",
             "모든 neural arm에 unreconciled / BottomUp / OLS / MinT-shrink를 동일하게 제공했습니다. Statistical baseline도 같은 48개월 context와 reconciliation을 사용했습니다.", ""]
    for out in [RESULTS] + ([stage_b] if has_b else []):
        raw = pd.read_csv(out / "RAW_SCORES.csv")
        level_scores = pd.read_csv(out / "LEVEL_SCORES.csv")
        effects = pd.read_csv(out / "SEED_EFFECTS.csv")
        resources = pd.read_csv(out / "RESOURCES.csv")
        group = raw.dataset.iloc[0]
        hier_levels = level_scores[level_scores.arm == "LORA_HIER"]
        controls = level_scores[level_scores.arm.isin(["LORA", "LORA_SELF", "LORA_POOL"])]
        level_effects = hier_levels.merge(controls, on=["dataset", "seed", "reconciliation", "level"], suffixes=("_hier", "_control"))
        level_effects["difference"] = level_effects.rmsse_hier - level_effects.rmsse_control
        level_effects["relative_difference"] = level_effects.rmsse_hier / level_effects.rmsse_control - 1
        level_effects = level_effects.rename(columns={"arm_control": "comparator"})[
            ["dataset", "seed", "comparator", "reconciliation", "level", "rmsse_hier", "rmsse_control", "difference", "relative_difference"]]
        level_effects.to_csv(out / "LEVEL_EFFECTS.csv", index=False)
        lines += [f"## {group} 결과", "", "아래 neural 값은 선택된 seed들의 평균입니다. Seed별 방향은 별도로 제시합니다.", ""]
        summary = raw.groupby(["arm", "reconciliation"])[["primary", "bottom_rmsse", "raw_mae"]].mean()
        lines += ["```text", summary.to_string(float_format=lambda x: f"{x:.6f}"), "```", ""]
        mint = raw[raw.reconciliation == "mint_shrink"].groupby("arm").primary.mean()
        unreconciled = raw[raw.reconciliation == "unreconciled"].groupby("arm").primary.mean()
        questions = [("LoRA 자체가 F0 대비", "LORA", "F0"), ("SELF가 LoRA 대비", "LORA_SELF", "LORA"),
                     ("POOL이 SELF 대비", "LORA_POOL", "LORA_SELF"), ("HIER가 SELF 대비", "LORA_HIER", "LORA_SELF"),
                     ("HIER가 POOL 대비", "LORA_HIER", "LORA_POOL"), ("HIER가 LoRA 대비", "LORA_HIER", "LORA")]
        for title, a, b in questions:
            diff = mint[a] / mint[b] - 1
            pre = unreconciled[a] / unreconciled[b] - 1
            lines.append(f"- {title}: MinT primary {diff:+.2%}, reconciliation 전 {pre:+.2%} (음수가 유리).")
        baseline = min(["SeasonalNaive", "AutoETS"], key=lambda a: mint[a])
        lines += [f"- 가장 낮은 통계 baseline primary는 {baseline} {mint[baseline]:.6f}; HIER는 {mint['LORA_HIER']:.6f}입니다.", "",
                  "Seed 효과: HIER − 비교 arm. 양수는 HIER의 오차가 더 큽니다.", "", "```text",
                  effects[effects.reconciliation.isin(["mint_shrink", "unreconciled"])][["seed", "comparator", "reconciliation", "primary_difference", "bottom_relative_difference"]].to_string(index=False, float_format=lambda x: f"{x:+.6f}"),
                  "```", "", "각 level과 raw MAE는 LEVEL_SCORES.csv / RAW_SCORES.csv, 각 origin은 ORIGIN_SCORES.csv에 보존했습니다.", "",
                  f"Neural fits {int((resources.kind == 'neural_fit').sum())}개, main updates {int(resources.updates.sum()):,}회; 기록된 fit/baseline 실행 시간 합 {resources.seconds.sum()/60:.1f}분.", ""]
        lines += ["MinT 이후 level별 HIER 효과(선택된 세 seed 평균, 음수가 유리):", "", "```text",
                  level_effects[level_effects.reconciliation == "mint_shrink"].groupby(["comparator", "level"])[["difference", "relative_difference"]].mean().to_string(float_format=lambda x: f"{x:+.6f}"),
                  "```", "", "전체 seed/level/reconciliation 효과는 LEVEL_EFFECTS.csv에 있습니다.", "",
                  f"이 선택된 모델들에서 HIER primary는 raw {unreconciled['LORA_HIER']:.6f}, MinT {mint['LORA_HIER']:.6f}입니다. 합계 일치 자체를 예측 성능 개선으로 해석하지 않았고, TEST를 보고 reconciliation을 변경하지 않았습니다.", ""]
    lines += ["## 후속 결정과 한계", "", "```json", json.dumps(decision, indent=2, ensure_ascii=False), "```", "",
              "TourismLarge Stage B를 실행했습니다. 구조와 arm별 LR은 Labour에서 고정했고 새 seed 92211/92212만 사용했습니다." if has_b else
              "Labour continuation 조건을 모두 충족하지 않아 TourismLarge 학습은 실행하지 않았습니다. TourismLarge 다운로드·감사는 Stage B 학습과 구분합니다.", "",
              "선택 seed는 개발 과정에 사용되었으며 두 repeat seeds를 별도로 보고했습니다. overlapping origins와 작은 seed 수를 고려할 때 이 결과로 보편적 우월성이나 통계적 유의성을 주장할 수 없습니다.",
              "`OPTIMIZATION_LIMIT_REACHED` 표시는 마지막 checkpoint의 validation이 계속 개선된 fit에만 남겼으며 학습을 연장하지 않았습니다.",
              "추가 LR/rank/adapter/graph/loss 탐색은 실행하지 않았습니다. 환경·구현 오류 기록은 IMPLEMENTATION_EVENTS.json에 있으며 과학적 결과와 분리했습니다.", "",
              "모든 fit과 TEST 평가가 끝난 뒤 continuation boolean의 JSON 직렬화 오류가 발생했습니다. 완료된 점수표에서 동일 조건을 재계산하고 Python bool로 저장해 복구했습니다. 재학습·추가 update는 0회입니다. 실행 당시 source seal은 보존했고, 재발 방지용 후처리 수정과 회귀 테스트는 POST_RUN_REPAIRS.json에 별도 기록했습니다.", ""]
    report_text = "\n".join(line.rstrip() for line in "\n".join(lines).splitlines()) + "\n"
    (RESULTS / "REPORT_KO.md").write_text(report_text, encoding="utf-8", newline="\n")
    (RESULTS / "FINAL_DECISION.md").write_text(f"# Final decision\n\n`{label}`\n\nLabour gate: `{decision['decision']}`\n\nStage B executed: {has_b}\n\n판단 근거와 seed/level 효과는 REPORT_KO.md와 SEED_EFFECTS.csv에 기록했습니다. 논문 PASS/FAIL을 선언하지 않습니다.\n", encoding="utf-8", newline="\n")
    checks = {}
    for out in [RESULTS] + ([stage_b] if has_b else []):
        budget = json.loads((out / "TRAINING_BUDGET.json").read_text())
        ledger = [json.loads(line) for line in (out / "UPDATE_LEDGER.jsonl").read_text().splitlines()]
        main = [r for r in ledger if r["kind"] == "main"]
        counts = Counter(r["fit_id"] for r in main)
        assert len(main) == budget["main_updates"] == budget["max_main_updates"]
        assert len(counts) == budget["fits_completed"] == budget["max_fits"]
        assert set(counts.values()) == {512}
        assert len({(r["fit_id"], r["update"]) for r in main}) == len(main)
        for name in ["CHECKPOINT_MANIFEST.json", "PREDICTIONS_MANIFEST.json"]:
            for item in json.loads((out / name).read_text()):
                assert sha256(ROOT / item["path"]) == item["sha256"]
        checks[budget["stage"]] = {"fits": len(counts), "main_updates": len(main), "unique_updates": True,
                                   "all_artifact_hashes_valid": True, "smoke_updates": sum(r["kind"] == "smoke" for r in ledger)}
    seal = json.loads((RESULTS / "PROTOCOL_SEAL.json").read_text())
    repair_path = RESULTS / "POST_RUN_REPAIRS.json"
    repairs = json.loads(repair_path.read_text())["files"] if repair_path.exists() else {}
    applied_repairs = []
    for path, expected in seal["code_sha256"].items():
        actual = sha256(ROOT / path)
        if actual != expected:
            assert path in ["src/hier_peft/screen.py", "src/hier_peft/report.py"]
            assert repairs[path]["before_sha256"] == expected and repairs[path]["after_sha256"] == actual
            applied_repairs.append(path)
    files = [p for p in RESULTS.rglob("*") if p.is_file() and p.name not in ["RESULTS_MANIFEST.json", "VERIFICATION.json"]]
    write_json(RESULTS / "RESULTS_MANIFEST.json", [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(files)])
    write_json(RESULTS / "VERIFICATION.json", {"status": "PASS", "timestamp": utc_now(), "stage_checks": checks,
               "sealed_code_unchanged": not applied_repairs, "executed_source_seal_retained": True,
               "documented_post_run_output_only_repairs": applied_repairs,
               "test_report_sha256": sha256(RESULTS / "PREFLIGHT_TESTS.xml"),
               "stage_b_executed": has_b, "continuation_decision": decision["decision"], "final_decision": label,
               "scientific_significance_claimed": False})
    print(label)


if __name__ == "__main__":
    finalize()
