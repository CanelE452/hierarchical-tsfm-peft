# 작은 K 연구: 안정화 이후에도 남은 정확도 문제

이 문서는 통합 보고서의 실제 독립 감사가 끝난 시점의 상태다. 이후 실행은 새 기록으로 보존한다. 최신 직접 지시 “상한내수행하지 말고 쭉해줘”에 따라 이전 자원/시간 상한 때문에 종료하지 않는다. **연구 목표는 아직 해결되지 않았다.** TEST·설치·큰 백본 학습·기존 작업 중지/재개·자동 Git 게시는 포함하지 않는다. 이전 승인·사용량·실패·보고서를 덮지 않는다.

## 지금 확인한 답

[확인] 과거로만 적합한 origin별 디코더의 큰 계수를 정규화하는 대조는 실제 수행했다. Robin의 선정 PCA Bolt 개발 정규화 MSE는 static .532063803 → unregularized .431921567 → TUNE 선정 shrinkage .378742147이었다. 원 채널 Raw .340401887까지의 차이는 남았다. FIT-target의 선정 shrinkage .394111414는 static .564240397보다 낮지만 기존 unregularized 개발 .373466779보다 높다. 그 개발 점수를 보고 rho를 다시 고르지 않았다. Beijing의 선정 PCA/target 개발 full-observed MSE .696671169/.668548248도 Raw .589575414보다 높았다.

[확인] 새 디코더에 맞춰 작은 native head8개를 실제 재학습했다. Robin 개발 PCA seed01/02 .603474794/.579036158, target .659218802/.610172053; Beijing .809982105/.793996461, .787425449/.779279771이었다. 같은 현재값을48회 반복하는 기준은 Robin1.308464073/Beijing .908441877이다. 학습 효과가 있지만 Raw보다 나쁘다. 이 제한된 affine head·600update 조건의 실패를 모든 native 학습의 불가능이나 Bolt 필수성으로 바꾸지 않는다.

[확인] 선정한 과거 디코더의 **complete-C 실수 출력 공간**에서 미래를 아는 최소값은 Robin PCA/target .194717889/.202114826, Beijing .363049891/.332386372였다. 실제 미래 E(Y)를 넣은 진단도 각각 .230315141/.253156642, .392531695/.337172217이었다. 이런 여지가 실제 과거 입력으로 예측 가능하다는 증명은 아니다. Beijing complete-C Raw .598972295와 full-observed Raw .589575414를 섞지 않는다. 이 minimum은 정규화 MSE·저장 D32를 실수로 해석한 공간에 한정하며 엄밀 FP32 전체 출력/MAE/물리 MSE의 하한이 아니다.

[판단] 현재 실패를 전부 “재구성만 학습해서” 또는 “새 D와 head가 불일치해서”로 설명하기 어렵다. 기존 actual-Y 목적 학습도 있었고 새 D의 matched 재학습도 시행했다. 표현이 현재 backbone에 적합한 미래 신호를 남기는지와 복원 공간·수치 안정성·실제 예측기의 오류를 나눠 확인해야 한다. 대조학습·동적 그래프가 반드시 필요하다는 증거는 아직 없다.

## 실제 전체 비용과 반복

선정 디코더의5-route F-only20조건/120측정, 새7-route matched28조건/196측정, 같은 프로토콜의 별도 반복28조건/196측정을 각각 보존했다. 매 callback 물리 causal fill→정규화→E→관측 과거 D fitting/구현된 진단/input binding hash→실제 F 또는 native→D/currentC anchor→물리 CPU 반환/sync를 계측했다. 모델 준비·파일/상태 metadata hash·guard·warmup·저장은 밖이다. 이 구현의 최소 production 비용이나 방법별 독점 RSS/VRAM이라고 쓰지 않는다.

```text
B8 전체 ms/window              Raw       PCA/Bolt  Target/Bolt  PCA/native01/02  Target/native01/02
Robin 첫 matched               4.3323    6.0386    6.1202       3.1250/3.0964    3.0998/3.2108
Robin 별도 반복                4.3103    6.1842    6.0397       3.0937/3.1134    3.2722/3.0541
Beijing 첫 matched            10.4450    4.9314    5.1125       2.0620/2.0030    2.0444/2.0190
Beijing 별도 반복             10.0667    5.0267    5.0282       1.9952/1.9940    2.0398/1.9190
```

Beijing 앞선 F-only Raw는5.4856ms로 달랐다. 두7-route 시행이 비슷하게 높았지만5-route와의 차이 원인은 미확정이다. 모든 시행·원 sample/outlier·두 비용 비율을 남기며 빠른 시행만 고르지 않는다. 첫/반복 cohort별 TUNE MSE·MAE 각각 Raw1.01배 이내와 B8 Raw 중앙값 비율 .90 이하를 함께 확인한 방법은 **각각0/12**다. 두 cohort를24개 새 방법 또는 독립 데이터 반복이라고 세지 않는다.

## 완료된 보고서와 실제 증거

- [통합 한국어 보고서:22CSV·6PNG6SVG](reports/shrinkage_matched_native_v1/REPORT_KO.md)
- [보고서 독립 PASS](audits/SHRINKAGE_MATCHED_NATIVE_REPORT_AUDIT_v1.json):35파일·295,303 primitive CSV fields·62개 MD 숫자 문단/비용 행·52local링크·6PNG6SVG·원 보고서SHA 확인. ROOT가6PNG를 직접 확인했다.
- [shrinkage 실제 결과](runs/local_decoder_shrinkage_v1/RESULT.json)·[독립 PASS](audits/LOCAL_DECODER_SHRINKAGE_AUDIT_v1.json)
- [새 선정 D 출력공간 진단](runs/selected_shrinkage_output_space_v1/RESULT.json)
- [matched head 실제 학습](runs/matched_shrunk_native_v1/RESULT.json)·[독립 PASS](audits/MATCHED_SHRUNK_NATIVE_AUDIT_v1.json)
- [5-route F-only 비용](runs/selected_shrinkage_cost_v1/RESULT.json)·[독립 PASS](audits/SELECTED_SHRINKAGE_COST_AUDIT_v1.json)
- [첫7-route 전체 비용](runs/matched_shrunk_native_cost_v2/RESULT.json)·[독립 PASS](audits/MATCHED_SHRUNK_NATIVE_COST_AUDIT_v2.json)
- [동일 프로토콜 반복](runs/matched_shrunk_native_cost_repeat_v1/RESULT.json)·[독립 PASS](audits/MATCHED_SHRUNK_NATIVE_COST_REPEAT_AUDIT_v1.json)
- [과거 local D 보고서](reports/local_past_decoder_v1/REPORT_KO.md)·[상수 규칙 보고서](reports/past_constant_gate_v1/REPORT_KO.md)·[초기 비교의 역사적 index](README_KO.md)

![디코더를 안정화한 실제 예측](reports/shrinkage_matched_native_v1/figures/02_actual_Bolt_decoder_comparison.png)

![현재값과 재학습 native 대조](reports/shrinkage_matched_native_v1/figures/03_frozen_and_matched_native.png)

![세 비용 시행을 보존한 결과](reports/shrinkage_matched_native_v1/figures/05_separate_actual_whole_cost_cohorts.png)

## 실패 복구와 다음 비교

첫 native-cost v1은 배열 manifest가 없는 physical alias를 엄격 loader에 넘겨0셀/0측정으로 실패했다. 같은 원 file/path/SHA의 canonical array manifest를 결속한 fresh v2만 실행했으며 timer helper·예측·공차·head·rho·원 source/result를 바꾸지 않았다. 첫 감사의 상대 경로 호출도 absolute 필수 helper와 충돌했다. 같은 감사 source에 절대 --cost를 넘긴 fresh 감사가 PASS했고 원 실패를 보존했다. [I/O 실패 동일성](audits/MATCHED_NATIVE_COST_INPUT_SCHEMA_FAILURE_IDENTITY_v1.json)·[감사 호출 실패 동일성](audits/MATCHED_COST_AUDIT_RELATIVE_PATH_FAILURE_IDENTITY_v1.json).

[판단] 다음 [원 채널 subset 계획](PLAN_RAW_CHANNEL_SUBSET_KO_v1.md)은 작은 K·과거 전용 배포·고정 F를 유지하면서, 원 채널을 섞지 않은 입력을 PCA/FIT-target와 같은 FIT 관측 과거 global LS decoder 규칙으로 비교한다. Subset은 FIT 과거의 복원 점수로 greedy 선정하며 forecastability나 최적 채널집합을 보장하지 않는다. 원 혼합 공간의 감독 차이도 유지한다. [전체 비용 사전 계획](PLAN_RAW_SUBSET_WHOLE_COST_KO_v1.md)은40셀/400samples이며 새 과학 결과와 독립 감사 이후 실행한다. **이 문서 시점에 새 subset 선정·F 추론·native 실제 학습·비용은0이고 SOURCE/합성 검산 준비중이다.**

기존 D1에서는 다른 FIT-RAW basis로 forecast→projection 순서를 이미 실제 비교했고 [저장 배열 감사](../../smallk_failure_followup_20261014/validation_v1/D1_ARRAY_AUDIT_v1.json)는 독립 GPU replay 없이 PASS였다. 현재 PCA/FIT-target·현재 D의 결과로 재사용하지 않는다. RF3는 nonlinear paired-reference residual이고 origin-reference helper는 합성 검사만 있었으므로, 고정 causal lagged linear D와 구별한다. 이 후속을 첫 관련 연구라고 주장하지 않는다.

[자원 상한 변경](EXECUTION_SCOPE_ADDENDUM_KO_v2.md)·[직접 지시와 권한 기록](AUTHORIZATION_SCOPE_v1.json)·[원 보호 기록을 유지한 manifest](CONTINUATION_MANIFEST_v1.json)·[계측snapshot v3](audits/RESOURCE_SNAPSHOT_20261015_v3.json). v3는70terminal receipts/당시active0·CPU user+system747.4375초·GPU process meterwall296.2052초·당시폴더1.424GB의 특정 시점 기록이다. GPU meterwall은 CUDA kernel 시간이 아니며 source 검토·일부 합성 fixture는 미계측이다. 이후 실행·감사를 포함하지 않는다. 새 결과는 로컬이며 main에 자동 게시하지 않았다.
