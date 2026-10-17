"""Package saved public aggregates after user stop; never load models or raw arrays."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


FIELDS = "comparison_id dataset model seed L H C K split metric_scale fit selection epoch updates exposures parameters MSE MAE status verification source_file source_sha256 scaler_sha256 target_sha256 mask_sha256 origins_sha256 checkpoint_sha256".split()
STATUS_FIELDS = "block status completed registered unit reason resume_requirement".split()
ENDPOINTS = [(0.0, 0, 0), (0.25, 156, 2481), (1.0, 622, 9923), (3.0, 1863, 29769)]
PARAMS = {"B0": 1008, "B2": 35717, "B2_CAP_POINTWISE": 35708}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_text(rows, fields):
    result = io.StringIO(newline="")
    writer = csv.DictWriter(result, fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return result.getvalue()


def build(repo, report):
    bindings = {}
    checks = []

    def bind(path):
        relative = path.relative_to(repo).as_posix()
        bindings[relative] = digest(path)
        return relative

    def load(path):
        bind(path)
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def source_fields(path):
        return {"source_file": bind(path), "source_sha256": digest(path)}

    def base(model, seed):
        return dict.fromkeys(FIELDS, "NA") | {
            "dataset": "jena", "model": model, "seed": seed,
            "L": 512, "H": 48, "C": 21, "K": 6,
            "parameters": PARAMS[model],
        }

    stop_path = report / "STOP_STATUS_PUBLIC.json"
    stop = load(stop_path)
    assert stop["status"] == "STOPPED_CONFIRMED"
    assert stop["scientific_campaign_complete"] is False
    assert stop["new_dispatch_allowed"] is False
    assert stop["automatic_resume_allowed"] is False

    old_path = repo / "reports/informs_ppt_ready_20261017/PPT_MAIN_TABLES.csv"
    bind(old_path)
    old_all = list(csv.DictReader(io.StringIO(old_path.read_text(encoding="utf-8-sig"))))
    old = [r for r in old_all if r["table_id"] == "Temporal_Context_Effect" and r["dataset"] == "jena"]
    assert len(old) == 4
    refs = {(r["method"], int(r["seed"])): r for r in old}
    reference = old[0]
    canonical = {key: reference[key] for key in ("scaler_sha256", "target_sha256", "mask_sha256", "origins_sha256")}
    assert all(all(r[key] == value for key, value in canonical.items()) for r in old)
    rows = []
    selected = {}
    for seed in (92601, 92602):
        for model in ("B0", "B2"):
            r = refs[model, seed]
            epoch = float(r["selected_epoch"])
            row = base(model, seed) | {
                "comparison_id": "SELECTED_A60", "split": "V_select",
                "metric_scale": "canonical_A60", "fit": "fresh_A60_reused",
                "selection": "A_stop_search_0_.25_1_3", "epoch": epoch,
                "updates": int(r["refit_updates"]), "exposures": int(epoch * 12020),
                "MSE": float(r["MSE"]), "MAE": float(r["MAE"]),
                "status": "EXISTING_RESULT_REUSED",
                "verification": "EXISTING_FP64_AUDITED",
                "checkpoint_sha256": r["checkpoint_sha256"],
            } | canonical | source_fields(old_path)
            rows.append(row)
            selected[model, seed] = row

        run = report / "runs" / f"cap_s{seed}"
        summary = load(run / "SEARCH_SUMMARY.json")
        curves = load(run / "SEARCH_CURVES.json")
        refit = load(run / "REFIT_SUMMARY.json")
        path = run / "SELECTED_V_RESULT.json"
        result = load(path)
        assert summary["status"] == "SEGMENT_COMPLETE"
        assert result["status"] == "COMPLETE_SELECTED_V" and result["model"] == "B2_CAP_POINTWISE"
        assert result["seed"] == seed and result["params"] == 35708
        assert refit["status"] == "SEGMENT_COMPLETE"
        assert refit["updates"] == 752 and refit["exposures"] == 12020
        assert result["selected_epoch"] == summary["selected_epoch"] == refit["selected_epoch"] == 1.0
        assert len(curves) == 4
        for curve, (epoch, updates, exposures) in zip(curves, ENDPOINTS):
            assert (curve["epoch"], curve["updates"], curve["exposures"]) == (epoch, updates, exposures)
            rows.append(base("B2_CAP_POINTWISE", seed) | {
                "comparison_id": "CAP_SEARCH_A_STOP", "split": "A_stop",
                "metric_scale": "search_A50", "fit": "A_fit_search",
                "selection": "SEARCH_DIAGNOSTIC_NOT_V", "epoch": epoch,
                "updates": updates, "exposures": exposures,
                "MSE": curve["astop_mse"], "status": "COMPLETE_NATIVE_SEARCH_SCORE",
                "verification": "NATIVE_NOT_FINAL_FP64", "scaler_sha256": summary["contract"]["scaler_sha"],
            } | source_fields(run / "SEARCH_CURVES.json"))
        fps = result["array_fingerprints"]
        expected = {"target_sha256": fps["target_normalized"]["sha256"], "mask_sha256": fps["observed"]["sha256"], "origins_sha256": fps["origins"]["sha256"], "scaler_sha256": result["scaler_sha256"]}
        assert expected == canonical
        assert result["observed"] == 43344 and result["origins"] == 43
        # Saved sufficient totals only: this is not raw-prediction FP64 re-scoring.
        for metric, total in (("MSE", "channel_sse"), ("MAE", "channel_sae")):
            macro = math.fsum(value / count for value, count in zip(result[total], result["channel_counts"])) / 21
            checks.append({"file": path.relative_to(repo).as_posix(), "check": f"saved_{metric}_totals", "absolute_difference": abs(macro - result[metric])})
            assert abs(macro - result[metric]) < 1e-12
        row = base("B2_CAP_POINTWISE", seed) | {
            "comparison_id": "SELECTED_A60", "split": "V_select", "metric_scale": "canonical_A60",
            "fit": "fresh_A60", "selection": result["selection"], "epoch": 1.0,
            "updates": refit["updates"], "exposures": refit["exposures"],
            "MSE": result["MSE"], "MAE": result["MAE"], "status": "COMPLETE_SELECTED_V_NATIVE",
            "verification": "NATIVE_NOT_FINAL_FP64", "checkpoint_sha256": result["artifact_sha256"],
        } | expected | source_fields(path)
        rows.append(row)
        selected["B2_CAP_POINTWISE", seed] = row

        sources = sorted((report / "same_update" / f"s{seed}").glob("*.json"))
        assert len(sources) == 24
        for path in sources:
            result = load(path)
            assert result["model"] in ("B0", "B2") and result["seed"] == seed
            assert result["status"] == "COMPLETE" and result["same_update"] is True
            endpoint = next(item for item in ENDPOINTS if item[0] == result["epoch"])
            assert (result["updates"], result["exposure"]) == endpoint[1:]
            row = base(result["model"], seed) | {
                "comparison_id": "SAME_UPDATE", "split": result["eval"], "metric_scale": result["metric_scale"],
                "fit": result["fit"], "selection": result["selection"], "epoch": result["epoch"],
                "updates": result["updates"], "exposures": result["exposure"],
                "MSE": result["MSE"], "MAE": result["MAE"], "status": "COMPLETE_NATIVE_SCORE",
                "verification": "NATIVE_NOT_FINAL_FP64", "scaler_sha256": result["scaler_sha256"],
                "checkpoint_sha256": result["checkpoint_sha256"],
            } | source_fields(path)
            if result["eval"] == "V_select":
                fps = result["array_fingerprints"]
                expected = {"target_sha256": fps["target_normalized"]["sha256"], "mask_sha256": fps["observed"]["sha256"], "origins_sha256": fps["origins"]["sha256"], "scaler_sha256": result["scaler_sha256"]}
                assert expected == canonical
                row |= expected
            rows.append(row)
        for epoch, updates, exposures in ENDPOINTS:
            rows.append(base("B2_CAP_POINTWISE", seed) | {
                "comparison_id": "SAME_UPDATE", "split": "V_select", "metric_scale": "canonical_A60",
                "fit": "A_fit_search", "selection": "NOT_SELECTED_COMMON_ENDPOINT",
                "epoch": epoch, "updates": updates, "exposures": exposures,
                "status": "PARTIAL_NO_SAVED_OUTPUT", "verification": "NA",
            })

    analysis = load(report / "ERROR_DECOMPOSITION_SUMMARY.json")
    analysis_audit = load(report / "ERROR_DECOMPOSITION_AUDIT.json")
    assert analysis_audit["status"] == "PASS_INDEPENDENT_SCALAR_FP64_RECOMPUTATION"
    cap_diagnostic = load(report / "diagnostics/CAP_INITIALIZATION_AUDIT_PUBLIC.json")
    pems_diagnostic = load(report / "diagnostics/PEMS_RESTORE_AUDIT_RESULT.json")
    assert cap_diagnostic["status"] == "PASS_CAP_INITIALIZATION_BOTH_SEEDS"
    assert pems_diagnostic["status"] == "PASS_BOTH_RESTORE_DIAGNOSTICS"
    status = [
        ("A_saved_error_decomposition", "COMPLETE_AUDITED", 4, 4, "saved_prediction_archives", "Existing Jena B0/B2 two seeds decomposed and independently checked.", "None; no rerun requested."),
        ("B_existing_models", "COMPLETE_REUSED", 4, 4, "model_conditions", "Existing B0/B2 selected A60 and search checkpoints reused without training.", "None; retain original comparison conditions."),
        ("B_CAP_search_refit_selectedV", "COMPLETE_NATIVE_NOT_FINAL_FP64", 2, 2, "model_conditions", "Both CAP seeds completed search0/.25/1/3 and selected fresh-A60 1epoch plus V inference.", "Independent raw-prediction audit still pending; not authorized to rerun now."),
        ("B_common_update_saved_scores", "PARTIAL_USER_STOP", 48, 72, "checkpoint_split_scores", "24 saved B0/B2 JSON per seed; no completed CAP same-update scores.", "Explicit future resume instruction plus skip-completed evaluator; do not reuse selected refit as common-update."),
        ("C_fixed_linear_probes", "NOT_STARTED_USER_STOP", 0, 6, "probes", "Probe phase never dispatched.", "Future explicit instruction and original full-history/mask/regularization contract."),
        ("D_PeMS_C32_resume_training", "NOT_STARTED_USER_STOP", 0, 2, "model_conditions", "Copy/next-update restore audit passed earlier; campaign resume/full V never dispatched.", "Original partial checkpoints retained; explicit instruction and a new valid execution window required."),
        ("D_new_full_pipeline_cost", "NOT_STARTED_USER_STOP", 0, 4, "family_batch_routes", "New PeMS B0/B2 batch1/8 costs were not measured.", "Execute only after models finish and no competing training; not requested now."),
        ("E_new_independent_FP64_closeout", "NOT_STARTED_USER_STOP", 0, 1, "campaign_closeout", "New CAP/common-update raw predictions were not independently re-scored; A audit is a separate completed prior stage.", "Future explicit read-only validation instruction; report packaging is not that scientific audit."),
        ("Independent_future_generalization", "PENDING_NO_NEW_DATA", 0, 1, "unused_future_data_contract", "User confirmed no new data. Protected T remains locked.", "New unused data and an approved frozen evaluation contract."),
    ]
    status_rows = [dict(zip(STATUS_FIELDS, row)) for row in status]
    selected_lines = ["seed   model                params  epoch updates exposures MSE          MAE          verification"]
    for seed in (92601, 92602):
        for model in ("B0", "B2", "B2_CAP_POINTWISE"):
            r = selected[model, seed]
            selected_lines.append(f"{seed}  {model:<20} {r['parameters']:>6}  {r['epoch']:>4g} {r['updates']:>7} {r['exposures']:>9} {r['MSE']:.9f}  {r['MAE']:.9f}  {r['verification']}")
    matched_lines = ["seed   epoch updates exposures B0 MSE      B2 MSE      B2-B0       CAP MSE"]
    for seed in (92601, 92602):
        for epoch, updates, exposures in ENDPOINTS:
            pair = {r["model"]: r for r in rows if r["comparison_id"] == "SAME_UPDATE" and r["seed"] == seed and r["epoch"] == epoch and r["split"] == "V_select"}
            b0, b2 = pair["B0"]["MSE"], pair["B2"]["MSE"]
            matched_lines.append(f"{seed}  {epoch:>4g}  {updates:>6} {exposures:>9} {b0:.9f} {b2:.9f} {b2-b0:+.9f} NA")
    deltas = []
    for seed in (92601, 92602):
        b0, b2, cap = (selected[model, seed] for model in ("B0", "B2", "B2_CAP_POINTWISE"))
        deltas.append(f"seed {seed}: B2 대비 CAP MSE 차이 {cap['MSE']-b2['MSE']:+.9f}, MAE 차이 {cap['MAE']-b2['MAE']:+.9f}; B0 대비 CAP MSE 차이 {cap['MSE']-b0['MSE']:+.9f}.")
    native_v = [r for r in rows if r["comparison_id"] == "SAME_UPDATE" and r["split"] == "V_select" and r["model"] in ("B0", "B2")]
    matched_win = sum(1 for r in native_v if r["model"] == "B2" and r["MSE"] < next(t["MSE"] for t in native_v if t["model"] == "B0" and t["seed"] == r["seed"] and t["epoch"] == r["epoch"]))
    markdown = """# B2 원인·적용 범위 검증: 사용자 중단 시점 결과

[확인] 이 캠페인은 사용자의 중지 요청으로 종료됐다. 이 중지 요청은 원 PLAN/DATA_SCOPE의 계속 실행 문구보다 우선하며 자동 재개하지 않는다. 학습·추론·비용 측정은 재실행하지 않고, 이미 저장된 공개 JSON/CSV만 모아 이 보고서와 그림을 생성했다. **전체 실험은 미완료이며, 새 결과의 독립 최종 FP64 검산도 미실행이다.** 완료된 오류 분해의 독립 검산과 새 CAP 결과의 native 집계 상태를 구분한다.

## 먼저 말할 수 있는 것

[확인] 기존 선택 모델 비교에서 Jena 두 seed 모두 B2의 MSE가 B0보다 낮다. 근접 용량의 CAP 두 seed는 학습·선택 refit·V 추론을 끝냈다. 아래 CAP 수치는 원 실행의 저장 집계값이며 최종 독립 예측 재채점은 받지 않았다. 공통 학습량 표에는 B0/B2만 있고 CAP의 빈칸은 그대로 남긴다.

[추정] 현재 결과는 명시적 시간 혼합이 도움이 될 가능성과 양립한다. 그러나 CAP 공통 학습량 평가, 동일 probe와 PeMS 검증이 끝나지 않아 시간 처리 자체가 원인이라고 확정할 수 없다. 모수 수가 비슷해도 연산·함수 공간·최적화 난이도가 같지는 않다.

[미완료] 표현의 선형 예측 유용성, PeMS C32의 B2 추가 가치, 새 전체 비용, 새로운 미래 구간의 일반화는 확인되지 않았다. 사용자가 새 자료가 없다고 답했으므로 보호 T를 열지 않았다. 충분 학습·수렴과 공통 요인 식별도 입증하지 않았다.

## 계산 구조와 비교 계약

주 방법은 `B2 시간·채널 P → E(C→K) → 동결 Chronos-Bolt → 잠재 과거·미래를 받는 lag D → 원 C 현재값 anchor`다. P/E/D는 재구성 오차가 아닌 미래 예측 MSE로 이미 학습한다. 따라서 현재 미확정 원인을 재구성 목적 실패로 설명하지 않는다. B0도 Bolt와 D를 통해 전체 과거를 사용한다.

Jena는 L=512, H=48, C=21, K=6, seed=92601/92602, CPU 실행이다. A_fit=[0,10482), A_stop=[10482,12579), fresh A60=[0,12579), 개발 V=[14675,16772)의 43 origin, stride=48, 관측 정답 43,344개다. 주지표는 관측된 각 채널의 SSE/count를 구한 뒤 활성 채널을 균등 평균한 정규화 MSE다. MAE도 같은 집계다. A_fit/A_stop 진단은 검색 A50 정규화, V는 공통 A60 정규화이며 다른 척도를 직접 비교하지 않는다. 학습 중 온라인 손실과 전체 A_fit checkpoint 재평가 손실도 다르다.

B2는 시간 MLP 512→32→512, 채널 MLP 21→32→21을 쓴다. CAP는 앞 시간 MLP를 각 시점에 공유하는 21→774→21로 바꾸고 뒤 채널 MLP·잔차·GELU·E/D·anchor를 유지한다. 활성 모수는 B2=35,717, CAP=35,708이며 더미 모수는 없다. 다만 전체 (L,C) 창의 LayerNorm 통계를 공유하므로 CAP를 완전한 시점 독립 모델이라고 부를 수 없다. 현재 C값 anchor는 압축을 우회해 보존된다.

## 동일 선택 규칙 이후 fresh A60 모델

검색 endpoint 0/.25/1/3epoch의 A_stop MSE로 학습 길이를 선택한 뒤 A60에서 새로 적합했다. 기존 B0/B2 네 조건은 재사용했고 CAP 두 조건만 새로 학습했다. 기존 B0는 3epoch, B2와 CAP는 1epoch를 선택했다. 이 표는 동일 update 비교가 아니다.

```text
{selected_table}
```

{deltas}

![선택 A60 모델별 기존·신규 결과와 검산 상태](figures/STOPPED_SELECTED_MODELS.png)

[벡터 원본](figures/STOPPED_SELECTED_MODELS.svg)

## 동일 update·동일 표본 노출량의 검색 checkpoint

아래는 A_fit 검색 가중치를 고정 endpoint에서 읽어 공통 A60 척도로 V를 평가한 값이다. 위 fresh A60 refit 표와 섞지 않는다. 두 seed에서 B0/B2의 4endpoint×3split=48개 JSON이 저장됐다. 등록된 72개 중 CAP 24개는 완료되지 않았다. CAP의 검색 A_stop 점수 8개는 CSV에 별도 `CAP_SEARCH_A_STOP`으로 보존했으며 V 빈칸을 채우지 않는다.

```text
{matched_table}
```

[확인] 같은 학습량의 저장 V 비교에서 B2 MSE가 B0보다 낮은 endpoint는 {matched_win}/8개다(0epoch 포함). 이는 native 평가 집계이며 새 독립 최종 FP64 재채점은 남아 있다. CAP 비교가 빠져 있으므로 근접 용량까지 통제된 세 구조의 결론은 미완료다.

![저장된 동일 학습량 점수와 CAP 미완료 표시](figures/STOPPED_MATCHED_UPDATES.png)

[벡터 원본](figures/STOPPED_MATCHED_UPDATES.svg)

## 기존 오차 분해에서 확인한 개선 위치

[확인] 선택 B2−B0의 전체 MSE 차이는 seed92601 −0.100661035, seed92602 −0.076194853이다. 개선 채널 수는 각각18/21, 14/21이며 두 seed 공통으로 개선된 채널은13개다. 상위3개 채널이 차지하는 총 양의 이득은34.1%/45.0%로, 순 이득이 아닌 악화 상쇄 전 총 양의 이득을 분모로 쓴다.

[확인] 장거리33–48의 MSE 차이는 −0.207601972/−0.147412831이다. 근거리1–16은 −0.002579452/+0.005703784로 seed에 따라 방향이 갈리고, 근거리 MAE는 두 seed 모두 악화된다. 따라서 모든 거리·채널에서 고르게 좋아졌다는 문장은 맞지 않는다. 개발 V를 첫/중앙/끝15/14/14 origin으로 나눈 분석은 같은 V의 부분 진단이다.

이 단계는 저장 예측4개에 대해 독립 scalar FP64 검산을 완료했다. 150행·1,808개 수치 등식의 최대 오차는1.1991e-14(허용1e-12), 전체 분모의 가산 기여 복원 최대 오차는4.72e-16이다. 이 검산 통과를 새 CAP/공통 update의 최종 검산 통과로 확장하지 않는다.

![채널별 기존 B2−B0](figures/A_CHANNEL_EFFECT.svg)

![거리·시간 블록별 기존 B2−B0](figures/A_HORIZON_BLOCK_EFFECT.svg)

## 완료·미완료와 중단 보존 상태

중단 확인 시각은 `{stopped_at}`다. 실제 실행 트리가 종료되고 새 dispatch·자동 재개가 차단된 상태다. CAP 학습은 이미 끝났고, 중단된 것은 공통 update 재평가 두 작업과 제어기다. 학습 optimizer update의 손실은 없으며 저장 전이던 평가 batch만 결과가 남지 않았다. 원 checkpoint·선택 가중치·optimizer/sampler/RNG와 저장 집계는 보존했다. 평가 CLI는 현재 skip-completed 재개 모드가 없어 향후 명시 재개 시 별도 검토가 필요하다.

- 완료: 기존 B0/B2 재사용4조건, CAP 검색·fresh refit·선택 V2조건, 기존 예측 오차 분해와 독립 검산.
- 부분 완료: 공통 update 평가는48/72개 checkpoint×split 집계; 근접 용량 CAP의24개 평가는 미완료.
- 미실행: 동일 선형 probe0/6, PeMS 실제 재개·최종 V0/2, 새 전체 비용0/4, 새 결과 독립 최종 FP64 closeout.
- 선행 감사만 완료: [CAP 초기 공통 상태·활성 모수/gradient 경로](diagnostics/CAP_INITIALIZATION_AUDIT_PUBLIC.json), [PeMS 복사본의 다음 update optimizer/sampler/RNG 동등성](diagnostics/PEMS_RESTORE_AUDIT_RESULT.json). PeMS의 실제 진단 update는 총4회·표본노출64회이며 전체 재개 학습과 별개다. 이들은 새 성능이나 PeMS 전체 학습 완료를 뜻하지 않는다.
- 미완료 일반화: 새 미사용 자료가 없고 보호 T는 잠금 상태다. 동일 개발 V와 두 seed는 독립 시간 일반화의 근거가 아니다.

![단계별 실제 완료와 미완료](figures/STOPPED_WORK_STATUS.png)

[벡터 원본](figures/STOPPED_WORK_STATUS.svg)

재개에는 사용자 명시 지시와 유효 자원·시간 창이 필요하다. 현재 보고서는 재개·재추론·재학습을 하지 않는다. 향후 공통 update 평가는 저장된48개를 건너뛰는 경로를 먼저 검토해야 하며, PeMS는 원 C32/K8 partial 저장점을 복사해 복원 검사를 유지해야 한다. C170/K43 결과를 C32의 빈칸에 넣지 않는다. 새 비용은 모델 종료 후 단독 CPU2 조건에서 측정해야 한다. 미실행 경로의 소요 자원은 실측되지 않아 현재 추가 소요 시간을 확정하지 않는다.

## 발표에 사용할 수 있는 문장과 한계

“반복 사용한 Jena 개발 구간의 두 seed에서 B2는 B0보다 낮은 선택 모델 MSE를 보였고, 이득은 장거리에서 컸다. 가까운 거리의 MAE는 오히려 악화됐다.” — 기존 저장 예측 분해와 독립 FP64 감사가 근거다.

“모수 수가 거의 같은 시점별 연산 대조를 실제 학습했으나, 세 구조의 동일 학습량 비교와 동일 probe·PeMS 검증은 사용자 중단으로 완성되지 않았다. 시간 혼합이 개선의 원인이라는 결론은 아직 유보한다.” — CAP2조건의 학습/선택 V 완료 및 common-update 빈칸이 근거다.

“이 결과로 중요한 공통 패턴의 식별, 독립 미래 일반화, Bolt의 필수성 또는 정확도·비용 동시 우위를 주장하지 않는다.” — probe·새 자료·새 비용의 부재와 기존 비교의 구조/용량 차이가 남아 있다. 전체 창 정규화, 현재값 anchor, 제한된3epoch 검색, 상류 보간 계보 UNKNOWN도 명시한다.

## 산출물과 재현 범위

- [모든 저장 비교값 CSV](COMPARISONS_AT_STOP.csv): `SAME_UPDATE`, `SELECTED_A60`, `CAP_SEARCH_A_STOP`을 별도 비교로 표시한다.
- [단계별 상태 CSV](STATUS_AT_STOP.csv), [공개 입력 SHA 결속 및 집계 점검](STOPPED_SOURCE_BINDINGS.json).
- [기존 오류 분해 상세](ERROR_DECOMPOSITION_KO.md), [기존 독립 감사](ERROR_DECOMPOSITION_AUDIT.json), [자료 사용 이력](DATA_SCOPE_KO.md), [등록된 원 계획](PLAN_KO.md).
- [원 실행 코드](code/mechanism_runner.py), [CAP 대조](code/cap_control.py), [선형 probe](code/linear_probe.py), [PeMS 복원](code/pems_resume.py), [비용](code/pems_cost.py), [중단 보고 생성기](code/build_stopped_report.py).

보고 생성기는 표준 라이브러리로 공개 JSON/CSV만 읽고 SHA·스키마·비교 조건·저장 충분통계의 산술 일관성을 확인한다. 원 예측 배열에서 오차를 다시 계산하지 않는다. 가중치 로드·학습·모델 추론·비용 측정·T 접근은 모두0회다. 원본을 덮지 않으려면 새 출력 폴더로 실행한다.

공개 코드는 실행·집계 경로를 보여준다. 원 자산과 전용 감사 harness는 비공개로 보존돼 있으므로 공개 코드·영수증만으로 전체 학습과 선행 감사의 완전한 fresh 재현이 가능하다고 주장하지 않는다.

```powershell
python reports/informs_b2_why_where_20261017/code/build_stopped_report.py --execute --repo . --output-dir reports/informs_b2_why_where_20261017_rebuild
```
""".format(selected_table="\n".join(selected_lines), matched_table="\n".join(matched_lines), matched_win=matched_win, deltas="\n\n".join(deltas), stopped_at=stop["stopped_at"])
    readme = """# B2 개선 원인·적용 범위: 중단 시점 공개 패키지

사용자의 중지 후 이미 저장된 코드·집계 결과를 정리한 패키지다. **실험 전체 미완료**이며 재학습·재추론·새 비용 측정을 하지 않았다. 새 CAP native 점수와 기존 독립 감사 완료 결과를 구분한다.

- [한국어 결과·한계·발표 문구](STOPPED_RESULTS_KO.md)
- [모든 비교값 CSV](COMPARISONS_AT_STOP.csv) / [완료·미완료 상태 CSV](STATUS_AT_STOP.csv)
- [공개 입력·생성 코드 SHA 및 점검](STOPPED_SOURCE_BINDINGS.json)
- [정지 상태 공개 영수증](STOP_STATUS_PUBLIC.json)
- [기존 오류 분해 상세·그림](ERROR_DECOMPOSITION_KO.md)
- [보고 생성 코드](code/build_stopped_report.py)

![선택 모델 결과](figures/STOPPED_SELECTED_MODELS.png)

![공통 학습량 비교와 미완료](figures/STOPPED_MATCHED_UPDATES.png)

![단계별 완료 상태](figures/STOPPED_WORK_STATUS.png)

`SAME_UPDATE`는 A_fit 검색 checkpoint, `SELECTED_A60`는 A_stop 선택 후 fresh A60 refit이다. CAP의 선택 모델 점수를 공통 update의 빈칸에 넣지 않는다. 독립 미래 일반화와 수렴은 미확정이며 보호 T는 잠금 유지다. 재현 명령과 계약은 상세 보고서에 있다. 원자료·예측 배열·체크포인트·개인 절대경로는 이 패키지에 포함하지 않는다.
"""
    bind(Path(__file__).resolve())
    binding = {
        "schema": "STOPPED_PUBLIC_AGGREGATION_BINDINGS_v1", "status": "PASS_SAVED_AGGREGATION_ONLY_CAMPAIGN_INCOMPLETE",
        "inputs": [{"path": path, "sha256": value} for path, value in sorted(bindings.items())],
        "aggregation_checks": checks,
        "new_training": 0, "new_model_inference": 0, "new_cost": 0, "raw_prediction_rescoring": 0,
        "protected_T_numeric_access": 0, "scientific_campaign_complete": False,
        "new_native_results_independent_FP64_audited": False,
        "existing_error_decomposition_independent_FP64_audited": True,
        "rows": {"comparison": len(rows), "status": len(status_rows), "saved_common_update": 48, "missing_common_update": 24},
        "canonical_Jena_V": canonical,
        "scope": "Public saved aggregates only; sufficient-total arithmetic checks do not establish raw-prediction FP64 correctness or independent generalization.",
    }
    return {
        "COMPARISONS_AT_STOP.csv": csv_text(rows, FIELDS),
        "STATUS_AT_STOP.csv": csv_text(status_rows, STATUS_FIELDS),
        "STOPPED_RESULTS_KO.md": markdown,
        "README_KO.md": readme,
        "STOPPED_SOURCE_BINDINGS.json": json.dumps(binding, ensure_ascii=False, indent=2) + "\n",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print("SOURCE_ONLY: no input read, no model/data access, no output written")
        return
    repo = args.repo.resolve()
    report = repo / "reports/informs_b2_why_where_20261017"
    output = args.output_dir.resolve() if args.output_dir else report
    contents = build(repo, report)
    existing = [name for name in contents if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Preserve existing outputs: {existing}; choose a new --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    for name, text in contents.items():
        with (output / name).open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
    print(json.dumps({"status": "PASS_SAVED_AGGREGATION_ONLY", "files": list(contents), "scientific_campaign_complete": False, "models_loaded": 0, "experiments_rerun": 0}))


if __name__ == "__main__":
    main()
