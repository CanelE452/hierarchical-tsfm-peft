# 작은 K 연구: 원 채널 선택 대조와 다음 디코더 실험

이 문서는 raw subset 묶음의 실제 실행·비용·보고서 검산 완료 시점의 기록이다. 최신 직접 지시 “상한내수행하지 말고 쭉해줘”를 따라 이전 자원 상한으로 종료하지 않고 계속한다. **정확도와 전체 비용을 함께 만족하는 연구 목표는 아직 해결되지 않았다.**

## 이번에 확인한 답

[확인] 원 채널을 섞지 않고 K개만 고르는 대조를 실제 수행했다. 과거 FIT 복원 점수의 greedy 선택은 Robin K4의 Lindsay/Saul/Victor/Wai, Beijing K2의 CO/DEWP였다. 최적 채널집합이나 미래 예측 목적 선정은 아니다. 같은 고정 FIT-past D 적합 규칙과 실제 K-stream Bolt의 개발 정규화 MSE는 다음과 같다.

```text
자료/기노출 개발 구간       Raw Bolt       PCA/Bolt     FIT-target/Bolt   Raw-subset/Bolt
Robin VAL                   .340401887     .532048935    .548051499        .812393864
Beijing EVAL                .589575414     .778238706    .696929506        .866360117
```

[판단] 혼합을 없애도 해결되지 않아 채널 혼합만을 유일 원인으로 볼 수 없다. Subset도 정보와 D 공간을 함께 바꾸므로 혼합의 인과 효과를 완전히 분리한 대조는 아니다. 실제 선택 채널의 새 K-F 출력과 기존 full-C 출력 slice 차이는 6개 split에서 최대 약 .0000024 이내였다. C 전체 복원의 문제와 선택된 K 시계열의 F 출력 자체를 구분해야 한다.

![실제 Bolt 예측](reports/raw_subset_comparison_v1/figures/01_actual_Bolt.png)

[확인] 새 D32에 맞춘 zero-init native 12조건도 actual-C 미래 MSE로 600update씩 새로 학습했다. 전체 TUNE으로 CP를 고른 뒤 개발 구간을 재생했다. 모두 Raw보다 낮은 개발 MSE를 얻지 못했다. 현재값 반복보다 좋아지는 학습 효과는 있었으므로 학습이 전혀 안 됐다는 뜻은 아니다. 기존 actual-Y 학습도 있었기에 지금까지의 실패를 재구성 목적 하나로 확정할 수 없다. 같은 affine class·학습 규칙의 결과이며 모든 native의 한계나 Bolt 필수성을 증명하지 않는다.

## 디코더 제약과 예측 오차

[확인] 각 새 고정 D32의 complete-C 미래 실수 출력공간 minimum을 별도로 다시 계산했다.

```text
자료                     같은 mask Raw       PCA D 최소      Target D 최소    Subset D 최소
Robin VAL                 .340401887          .396541617      .392970126       .413679231
Beijing EVAL              .598972295          .511797659      .397358393       .537346126
```

[판단] Robin은 이번 세 시점별 고정 D 공간을 그대로 두는 것 자체에 제약이 있다. Beijing에는 공간의 여지가 있지만 실제 F와 새 native가 도달하지 못했다. 낮은 oracle는 과거 예측성을 뜻하지 않는다. E(Y)도 minimum이 아니며, 특히 subset의 E(Y)는 Robin .673077220/Beijing .657764618이었다. 이 수치는 정규화 MSE의 해당 수치 실수 공간 진단이다. 엄밀한 모든 FP32 출력 하한·MAE/물리 MSE 최소·인과 실패 비율로 부르지 않는다. Beijing full-observed Raw와 complete-C Raw를 섞지 않는다.

![새 고정 D의 출력공간 진단](reports/raw_subset_comparison_v1/figures/03_new_fixedD_diagnostic.png)

## 실행·비용·보고 상태

- [전체 한국어 보고서:19CSV·6PNG6SVG](reports/raw_subset_comparison_v1/REPORT_KO.md)
- [실제 과학 결과](runs/raw_subset_comparison_v2/RESULT.json)·[독립 과학 PASS](audits/RAW_SUBSET_COMPARISON_AUDIT_v3.json)
- [실제 전체 비용:40조건·400표본](runs/raw_subset_cost_v1/RESULT.json)·[독립 비용 PASS](audits/RAW_SUBSET_COST_AUDIT_v2.json)
- [보고서 독립 PASS](audits/RAW_SUBSET_REPORT_AUDIT_v1.json):32파일·171,848 primitive CSV fields·47상대링크·6PNG6SVG를 검산했다. ROOT는 그림 여섯 개를 직접 확인했다.
- [보고서 파일 manifest](reports/raw_subset_comparison_v1/MANIFEST.json)·[독립 검산 전 생성 영수증](audits/RAW_SUBSET_REPORT_GENERATION_v1.json)
- [앞 shrinkage·matched-native 통합 보고서](reports/shrinkage_matched_native_v1/REPORT_KO.md)·[그 이전 상태 기록](CONTINUATION_STATUS_KO_v3.md)

실제 새 subset F는 FIT/TUNE/DEV 6개 split/103model batch였고 기존 mixed F 12개 cache는 정확한 입력 lineage일 때만 재사용했다. 새 native 12조건/7200update/60CP와 원 실패 v1의 mixed8조건/4800update/40CP를 구분한다. 두 시행의 mixed8은 CP·곡선·예측·선정이 수치 공차 없이 동일했다. 실제 학습20회와 서로 다른 조건12개를 혼동하지 않는다.

비용은 원단위 causal fill·FIT normalize→E→실제 F 또는 native→D/currentC anchor→원단위 CPU 반환/sync까지다. 이번 D는 FIT 고정이므로 origin마다 적합하지 않는다. D/E/head 준비·hash·warmup·저장·guard·memory snapshot은 밖이다. 같은 첫 B개 origin의 반복이며 독립 데이터400개가 아니다.

```text
B8 전체 ms/window       Raw       PCA F     Target F    Subset F
Robin                   5.5042    3.3109    3.4544      3.2791
Beijing                 3.5567    3.1828    3.2103      3.2206
```

비율은 같은 시행의 Raw로 계산하고 paired ratio 중앙값과 ratio-of-medians를 별도 보존한다. TUNE MSE·MAE가 각각 Raw×1.01 이하이고 B8 ratio-of-medians≤.90인 동시 기준 통과는 **0/18**이다. 빠르다는 사실만으로 채택하지 않는다. 공유 resident process의 RSS/VRAM은 방법별 독점 메모리가 아니며 절대 시간 변동의 원인은 확정하지 않았다. 이전 dynamic-D 비용과 같은 경로가 아니다.

![전체 경로 실제 비용](reports/raw_subset_comparison_v1/figures/06_whole_cost.png)

## 원 실패 보존과 다음 실행

최초 subset score는 Gram 상쇄로 음수 MSE가 나와 [v1 부분 결과](runs/raw_subset_comparison_v1/RESULT.json)를 보존했다. [직접 잔차 진단](audits/RAW_SUBSET_GRAM_FAILURE_DIAGNOSTIC_ROOT_v1.json)·[직접 LS 복구](RAW_SUBSET_GRAM_NUMERIC_REPAIR_KO_v2.md)는 음수 clipping이나 tolerance 확대가 아니다. 과학 감사의 bool/mask 변수 충돌과 비용 감사의 Markdown/JSON 형식 오류도 원 실패를 남기고 각각 새 감사에서만 수정했다. [과학 감사 복구](RAW_SUBSET_AUDITOR_CONTROL_FLOW_REPAIR_KO_v3.md)·[비용 감사 복구](RAW_SUBSET_COST_AUDITOR_FORMAT_REPAIR_KO_v2.md).

[판단] 다음 [고정 causal lag·학습 목적 2×2 계획](PLAN_CAUSAL_LAG_OBJECTIVE_KO_v1.md)은 현재 E/K/F 캐시를 유지해 시점별 D 대 지연 D, 과거 재구성 D 대 FIT 예측 보정 D를 교차 비교한다. 원시 그래프/대조학습/큰 모델을 추가하기 전에 고정 D의 구조와 anchored 예측 목적 불일치를 확인한다. 현재 시점에 이 새 24 D·72예측은 미실행이고 SOURCE 준비중이다. 기존 D2에는 예측 목적 decoder refit이 이미 있었으며 이번을 최초 구현이나 새 구조 발견으로 부르지 않는다. 동일 lag의 배포 용량은 같지만 두 목적의 target 시점과 감독 정보는 다르고, lag 간 slope 수는 7배다.

[현재 자원 snapshot v4](audits/RESOURCE_SNAPSHOT_20261015_v4.json)는83terminal receipts/당시RUNNING0·CPU user+system999.359375초·GPU process meterwall409.558804초·폴더1,582,514,989bytes를 기록한다. GPU meterwall은 CUDA kernel 시간이 아니며 source 검토·일부 fixture/메타데이터 작업은 미계측이다. 원32legacy receipts/18보호기록을 유지했다. [최신 권한](AUTHORIZATION_SCOPE_v1.json)·[원 상한 보존 manifest](CONTINUATION_MANIFEST_v1.json).

모든 후속은 기노출 개발 자료의 탐색이다. TEST·설치/다운로드·큰 백본 학습·기존 중단 작업 재개/종료·자동 commit/push·논문/PPT 작성은 하지 않았다. 새 결과는 로컬이다. Git HEAD는 da1114945f90b32017b3caf39b81538557af6fd1이며 main 게시를 완료했다고 주장하지 않는다.
