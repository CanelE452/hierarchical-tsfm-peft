# 계속 실행 현황 v5 — 디코더 시간 구조와 학습 목적 대조 완료

[확인] 현재 세 고정 표현·작은 K·동결 Bolt에서 디코더를 시간적으로 확장하거나 예측 목적으로 적합하는 것만으로는 Raw의 정확도에 도달하지 못했다. 실제 24개 D·72개 split 예측, 별도 전체 비용52조건·676표본, 한국어 보고서와 독립 검산을 마쳤다. **전체 연구가 해결됐다는 뜻은 아니다.** 다음 선형 E 고정/갱신 대조는 SOURCE 준비 중이며 실제 probe·새 학습은 아직 0이다.

## 무엇을 확인했는가

- 정보 사용: FORECAST D는 FIT 미래 정답을 손실 target으로 쓰는 정상 감독 학습이다. 배포 특징은 과거와 실제 동결 F 예측뿐이다. 24개 D 잠금 후 이번 TUNE·DEV를 열었다. 72개 label/mask perturbation은 고정 캐시·D의 좁은 검사이며 원시 자료부터 전체 누출 검사를 재실행한 결과나 독립 일반화 증거가 아니다.
- 시간 구조: Robin/PCA의 과거 재구성 D는 lag0→lag7에서 개발 MSE .534928482→.488093014로 낮아졌지만 Raw .340401887보다 크다. 같은 complete-C 유지 수치 공간 minimum은 .400907405→.306885041이었다. 일부 출력 공간 제약을 완화하는 것과 과거로 적절한 잠재 미래를 예측하는 것은 다르다. 이 실험은 표현 E를 새로 학습한 비교도, 채널 간 인과 지연 그래프 검증도 아니다.
- 목적: 예측 목적 D가 같은 lag의 재구성 D보다 MSE를 줄인 비교는 TUNE12/12, 노출된 DEV5/12였다. 목적 변경의 이득이 개발 구간까지 일관되게 이어지지 않았다. 두 목적의 target 시점·분포·감독 정보가 다르며 lag7의 계수는 7배다.
- 남은 문제: 실제 잠재 미래와 새 D의 적합 문제, 표현 적응 가능성을 분리해야 한다. 낮은 미래 진단 minimum이나 E(Y)는 배포 성능·예측 가능성의 증거가 아니다. 현재 입력은 K streams와 과거 buffer·현재 C anchor여서 strict K-only가 아니다.
- 대조학습·그래프: 이번 결과만으로 필요성을 입증하지 않는다. 기존 D2·joint-Y·RF3·native 비교도 보존하며, 이미 실행한 F-gradient 표현 학습을 미실행 방법으로 소개하지 않는다.
- Bolt: 작은 K 경로의 비용 절감은 측정했지만 정확도·비용 동시 기준은0/24다. 현재로서는 채택 근거가 충분하지 않으며 작은 실제 표현 적응 대조로 남은 여지를 확인한다.

## 원수치와 비용

아래는 full-observed channel-macro 정규화 MSE다. Beijing의 complete-C 공간 진단과 섞지 않는다. 네 D와 세 E의 전체 원수치는 보고서 CSV에 있다.

```text
자료       E       D                 TUNE          노출된 DEV
Robin      PCA     L0_RECON         .103955141      .534928482
Robin      PCA     L7_RECON         .102808326      .488093014
Robin      PCA     L0_FORECAST      .086375610      .590244433
Robin      PCA     L7_FORECAST      .087457541      .594049759
Robin      Raw                     .056951091      .340401887
Beijing    Target  L7_FORECAST      .927140783      .657737024
Beijing    Raw                     .758435638      .589575414
```

![같은 E와 F의 구조·목적 비교](reports/causal_lag_objective_v1/figures/01_actual_accuracy.png)

비용은 physical causal fill/FIT normalize→실제 E→GPU Bolt→고정 lag D→물리 단위 FP32 CPU 반환/sync의 새 동일 cohort다. B8 Raw window당 Robin6.3661625ms/Beijing4.9106375ms이며 작은 K 경로는 빨랐다. ratio-of-medians와 paired ratio 중앙값을 각각 보존했다. 반복676개는 독립 데이터가 아니며 여러 E/D와 F가 함께 resident인 RSS/VRAM은 방법별 독점 메모리가 아니다. 정확도·비용 동시 기준은 TUNE MSE·MAE 각각 Raw×1.01 이하와 B8 ratio-of-medians≤.90이며 통과0/24다.

![정확도·비용 동시 기준](reports/causal_lag_objective_v1/figures/06_accuracy_cost_gate.png)

- [한국어 보고서·18CSV·6PNG6SVG](reports/causal_lag_objective_v1/REPORT_KO.md)
- [실제 과학 결과](runs/actual_causal_lag_objective_v1/RESULT.json)·[독립 과학 PASS](audits/CAUSAL_LAG_OBJECTIVE_AUDIT_v2.json)
- [실제 전체 비용](runs/actual_causal_lag_cost_v1/RESULT.json)·[독립 비용 PASS](audits/CAUSAL_LAG_COST_AUDIT_v1.json)
- [보고서 독립 PASS](audits/CAUSAL_LAG_REPORT_AUDIT_v1.json):31파일·967,908 중첩 CSV primitive fields·56상대링크·6PNG6SVG를 검산했다. ROOT는 PNG6개를 직접 확인했다.
- [보고서 manifest](reports/causal_lag_objective_v1/MANIFEST.json)·[생성 영수증](audits/CAUSAL_LAG_REPORT_GENERATION_v1.json)
- [이전 raw subset 결과 및 상태](CONTINUATION_STATUS_KO_v4.md)·[이전 보고서](reports/raw_subset_comparison_v1/REPORT_KO.md)

## 원 실패와 다음 실행

최초 공동 감사의 LS rank가 exact nonzero-column rank 상한을 초과한 문제는 원 실패를 보존하고 정확0열만 제거하는 대수 동등한 검산 경로로 복구했다. D·예측·cutoff·공차를 바꾸지 않았다. [최초 실패](audits/CAUSAL_LAG_OBJECTIVE_AUDIT_v1.json)·[ROOT 독립 rank 진단](audits/CAUSAL_LAG_RANK_BOUNDARY_DIAGNOSTIC_ROOT_v1.json)·[복구 설명](CAUSAL_LAG_JOINT_AUDITOR_ZERO_COLUMN_REPAIR_KO_v2.md). retained numeric minimum은 모든 실수/FP32 출력의 보편적 하한이 아니다. 미래 잠재 자유변수는 HK이며 관측 없는 horizon의 미래 변수도 유지했다.

다음 [선형 E 고정/갱신 계획](PLAN_LINEAR_E_ADAPTATION_KO_v1.md)은 동일 초기 PCA·FORECAST-LS에서 L0/L7×E고정/학습×2자료×2 batch-order seed의16조건을 비교한다. Torch CP0·기존 NumPy와의 parity, F를 통한 E gradient와 직접 과거 gradient를 먼저 분리 확인하고 본 학습600update·TUNE 선택·모든 잠금 후 DEV를 진행한다. E 학습이 좋아져도 F-gradient 필요성이나 Bolt 필요성을 단정하지 않는다. 현재 해당 probe·9,600 본 학습 update는 미실행이다.

[자원 snapshot v5](audits/RESOURCE_SNAPSHOT_20261015_v5.json)는91terminal receipts/당시RUNNING0·서로 다른91process group·CPU user+system1,298.15625초·GPU process meterwall464.434244초·폴더2,128,838,934bytes를 기록한다. 원32legacy receipts와18보호 기록은 유지했다. GPU meterwall은 kernel 시간이 아니며 SOURCE 검토·일부 fixture/metadata 작업은 미계측이다. 최신 상한 해제는 [권한](AUTHORIZATION_SCOPE_v1.json)·[원 기록 보존 manifest](CONTINUATION_MANIFEST_v1.json)에 반영되어 있다.

기노출 두 자료와 제한된 K의 적응적 탐색이다. TEST·설치/다운로드·큰 백본 파라미터 학습·기존 작업 종료/중단 실험 재개·자동 commit/push·논문/PPT 작성은 하지 않았다. HEAD da1114945f90b32017b3caf39b81538557af6fd1 그대로이며 이 후속 결과는 로컬이다.
