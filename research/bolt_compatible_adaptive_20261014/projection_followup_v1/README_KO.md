# 작은 K 표현·예측·복원 진단의 현재 결과

사용자의 “상한내수행하지 말고 쭉해줘”에 따라 이전 시간·누적 자원 상한 때문에 멈추지 않고 후속을 수행하고 있다. **연구 목표는 아직 해결되지 않았다.** 아래 완료는 특정 실행·검산의 완료이며 정확도 개선이나 독립 일반화 검증을 의미하지 않는다. 원 승인·자원 기록과 실패한 시행을 보존한다. 보호 TEST·대형 백본 재학습·기존 작업 종료/재개·설치·자동 commit/push는 하지 않는다. 이 폴더의 새 결과는 로컬 결과다.

## 현재 답과 남은 판단

- **고정 복원 공간이 실제 병목일 수 있다.** 지정한 static D의 실수 선형 공간 안에서 미래 정답을 아는 최적 잠재값을 허용해도 Robin PCA/ForeCA와 Beijing ForeCA는 원 채널 Bolt의 개발 MSE에 도달하지 못했다. 이것은 해당 D의 한계다. 다른 공간·조건부 D·비선형 복원이 모두 불가능하다는 증거는 아니다.
- **작은 K 자체의 불가능성은 입증되지 않았다.** 각 split의 실제 미래 전체를 아는 별도 rank-K oracle는 Robin 개발 .117363274, Beijing .349213093이었다. 미래 정답으로 공간을 만들고 잠재값도 아는 진단이므로 배포 방법·예측 가능성·학습 성능으로 사용할 수 없다.
- **단순 inverse decoder 교체는 해결하지 못했다.** E와 예측한 잠재 미래를 고정하고 D를 pinv(E)로 바꾼 실제 개입에서 Robin ForeCA Bolt MSE가 .822981362에서 1.210370368로 나빠졌다. 출력 공간·scale·obliqueness가 함께 달라져 어느 한 요인의 인과 효과라고 단정하지 않는다.
- **FIT 미래 에너지에 맞춘 공간도 충분하지 않았다.** 같은 inverse D 규칙의 PCA 대조와 FIT-target 공간을 비교했다. Beijing의 개발 complete-C 출력 공간 최소 MSE는 .512486320에서 .386059363으로 낮아졌지만 실제 Bolt full-observed MSE는 .694519401로 원 채널 .589575414보다 나빴다. Robin에서는 FIT-target 개발 최소 MSE .422495770 자체가 원 채널 .340401887보다 컸다. 큰 미래 분산은 과거에서 예측할 수 있는 정보와 다르다.
- **과거에 상수인 채널로 퍼지는 오차는 일부 확인했다.** 관측 과거만 보고 출력을 현재값으로 유지하는 규칙은 Robin PCA Bolt 개발 MSE .532063803→.522997964, FIT-target Bolt .564240397→.553518497이었다. 원 채널 .340401887과의 차이는 남았다. 과거 상수여도 실제 미래가 변한 사례가 있어 보편적인 persistence 규칙으로 해석하지 않는다. C별 조건 정보를 디코더에 추가하므로 strict K-only 비교도 아니다.

원 채널 Bolt를 유지할 최종 근거가 확보된 것은 아니다. 현재 압축 경로의 정확도와 전체 비용이 원 채널 대조보다 함께 나아져야 하며, native 비교와 복원 원인 분리를 함께 보아야 한다. 기존 actual-Y 목적 학습도 있었으므로 현재 실패를 전부 “재구성만 학습했기 때문”이라고 설명할 수 없다. 그래프·대조학습·ExoTimer 방식의 필요성도 이 결과만으로 확정되지 않는다.

## 비교 조건

Robin C17/K4, Beijing Aotizhongxin C11/K2, 과거512·미래48, FIT256·TUNE64, 이미 노출된 개발 Robin30/Beijing146 origins다. 원 FIT scaler·causal forward-fill·true observed mask·origin·frozen Bolt를 유지한다. 주지표는 관측 채널을 균등하게 보는 정규화 MSE이며 MAE와 물리 단위도 저장한다. 개발 결과를 본 뒤 설계한 탐색적 후속이므로 새 독립 holdout 검증이 아니다.

Beijing의 full-observed 원 채널 개발 MSE .589575414와 complete-C .598972295는 평가 mask가 다르다. Static 출력 공간의 최소값·E(Y) oracle는 complete-C 조건으로 비교한다. 이 두 원 수치를 섞어서 개선율을 계산하지 않는다. 실제 배포 인터페이스는 K streams와 current C anchor를 사용한다.

## 완료된 실행과 검산 가능한 보고서

1. [PCA/ForeCA·Bolt/native·전체 비용 비교](reports/matched_projection_v1/REPORT_KO.md): 투영4개, distinct native8조건×600 update, checkpoint40개, 전체 비용20조건·120회. [실제 결과](runs/actual_projection_v2/RESULT.json), [독립 수치 감사 PASS](audits/PROJECTION_RESULTS_AUDIT_v2.json), [보고서 검산 PASS](audits/REPORT_FINAL_VERIFICATION_v1.json). 6PNG·6SVG.
2. [미래를 아는 rank-K 한계 진단](reports/oracle_rank_addendum_v2/REPORT_KO.md): 실제 미래 행렬6개, 배포·성능 선정에서 제외. [결과](runs/oracle_rank_ceiling_v1/RESULT.json), [독립 감사 PASS](audits/ORACLE_RANK_CEILING_AUDIT_v1.json), [보고서 검산 PASS](audits/ORACLE_RANK_REPORT_FINAL_VERIFICATION_v2.json). 2PNG·2SVG.
3. [D-only inverse 교체와 실제 전체 비용](reports/decoder_metric_intervention_v2/REPORT_KO.md): cached 잠재 미래에 대한 복원96건·3168score field, 새 전체 비용20조건·120회. [복원 감사 PASS](audits/DECODER_METRIC_PROBE_AUDIT_v1.json), [비용 감사 PASS](audits/DECODER_METRIC_COST_AUDIT_v1.json), [보고서 검산 PASS](audits/DECODER_METRIC_REPORT_FINAL_VERIFICATION_v2.json). 3PNG·3SVG.
4. [FIT 실제 미래 공간과 같은 D 조건의 PCA 비교](reports/FIT_target_comparison_v1/REPORT_KO.md): 새 target-space 적합2개, PCA 대조2개, fresh native8조건×600 update·checkpoint40개, 새 전체 비용20조건·120회. [결과](runs/actual_fit_target_v1/RESULT.json), [독립 감사 PASS](audits/FIT_TARGET_COMPARISON_AUDIT_v2.json), [보고서 검산 PASS](audits/FIT_TARGET_REPORT_FINAL_VERIFICATION_v1.json). 5PNG·5SVG·12CSV.
5. 과거 상수 채널 규칙: [실제 CPU 결과](runs/past_constant_gate_v1/RESULT.json)·[독립 감사 PASS](audits/PAST_CONSTANT_GATE_AUDIT_v1.json), 자료/split6개·방법42쌍·84배포 예측 배열·2016score field. [실제 전체 비용](runs/past_constant_gate_cost_v1/RESULT.json)·[독립 감사 PASS](audits/PAST_CONSTANT_GATE_COST_AUDIT_v2.json), Bolt6경로·24조건·144회. 통합 시각 보고서는 준비 중이다. 비용은 첫8 origins에서 측정했고 Beijing은 규칙 적용 대상이0개였으므로 gated timing 차이를 출력 변경의 속도 효과라고 설명하지 않는다. Native 비용은 이 묶음에서 측정하지 않았다.

![동일 FIT-target 조건의 실제 개발 오차와 고정 출력 공간](reports/FIT_target_comparison_v1/figures/02_output_space_and_forecaster_gap.png)

이미지 파일명·상대 링크는 실제 보고서 manifest 기준으로 확인하며, 새 보고서가 준비되면 다음 기록에 연결한다.

## 다음 실행

[관측 과거로 적합하는 local D 계획](PLAN_LOCAL_PAST_DECODER_KO_v1.md)·[순수 수학 모듈](local_past_decoder_v1.py)은 SOURCE·합성 검산 단계다. 각 origin의 실제 관측 과거에서만 centered truncated LS로 D를 적합하고 cached 잠재 미래와 원 selected head는 고정한다. **현재 이 README 기록 시점에는 실제 local D 실행 결과가 없다.** K streams·current C에 C 과거로 만든 origin별 K×C coefficients가 추가되므로 디코더 정보 조건의 변화와 전체 적합 비용을 별도로 보고해야 한다. 원 static-D/global rank-K 하한을 이 조건부 D의 하한으로 재사용하지 않는다.

## 보존된 실패와 해석 제한

- 첫 projection 시행은 checkpoint 저장 batch 불일치로 감사 실패했다. 새 producer v2에서 같은 batch32 계약으로 저장하여 검산했고 투영·40checkpoint·학습 curve·8승자가 v1과 같음을 확인했다. 두 시행의16 native 학습을16개 독립 조건이라고 세지 않는다.
- FIT-target 감사 v1은 NumPy bool의 JSON 직렬화 실패로 완전 artifact를 만들지 못했다. [실패 보존 기록](audits/FIT_TARGET_AUDIT_SERIALIZATION_FAILURE_IDENTITY_v1.json)과 새 v2의 Python bool 변환만 남겼으며 학습 재실행·수치 허용오차 변경은 없었다.
- 상수 규칙 비용 감사 v1은 감사자의 날짜 seed 오타로 실패했다. [실패·수정 동일성 기록](audits/PAST_CONSTANT_COST_SEED_FAILURE_IDENTITY_v1.json), seed 한 곳을 복구한 v2 PASS를 사용한다. 실제 timer를 다시 실행한 결과가 아니다.
- 학습 seed는 zero-init head의 minibatch 순서를 바꾼다. 비용은 한 시행의 반복 측정이며 공유 RSS/VRAM을 방법 고유 메모리로 해석하지 않는다. 미래 정답을 아는 oracle·부분 입력 perturbation 검사·합성 수학 fixture는 성능 개선이나 독립 일반화의 증거가 아니다.

[상한 변경 기록](EXECUTION_SCOPE_ADDENDUM_KO_v2.md)·[직접 사용자 지시](AUTHORIZATION_SCOPE_v1.json)·[원 자원 기록을 보존한 manifest](CONTINUATION_MANIFEST_v1.json)·[자원 관측 snapshot](audits/RESOURCE_SNAPSHOT_20261015_v1.json)을 함께 남긴다. 검산 가능한 원수치가 없는 경로는 완료라고 쓰지 않는다.
