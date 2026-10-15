# 같은 선정 보정의 오차항 분해

새 architecture나 추가 선정이 아니라 [시간 보정24조건](runs/horizon_decoder_diagnostic_v1/RESULT.json)의 저장된 baseline과 선정 예측을 같은 관측 mask에서 검산하는 진단이다. 원 결과 SHA는 `5f158382640dac74ec63cc3d1005b385c898821d3c032d51b543f2c9b27d7dac`다. 과학 감사 PASS 뒤 실제 배열을 읽는다. 새 fit/F/모델/GPU/alpha·arm·DEV winner 선택은0이며 승인 대기를 만들지 않는다.

각 FIT/TUNE/노출 DEV에서 실제 FP32 baseline b와 선정 FP32 예측 p를 그대로 FP64로 해석한다. 관측된 정답 y만 사용하며 `e0=b-y`, `c=p-b`, `e1=p-y`로 정의한다. c는 실수상 feature×계수 자체가 아니라 **실제로 반환된 FP32 예측의 변화량**이다. 따라서 재결합/반올림 오차를 숨기지 않고 다음 항등식을 같은 관측 집합에서 검산할 수 있다.

```text
MSE(p)-MSE(b) = 2 mean(e0*c) + mean(c*c)
             = interaction_term + correction_energy
```

각 채널의 true-observed 수로 먼저 평균한 뒤 active C를 균등 평균한다. 전체/각48h/사전1–16·17–32·33–48 구간을 각각 자체 관측 분모로 계산한다. 전체 macro를 h별 macro의 단순평균으로 재구성하지 않는다. 원 field의 score와 새 독립 항 계산의 오차·closure residual을 기록한다. 누락 채널/구간은 NA이며0으로 보충하지 않는다.

24조건×3split=72전체 cell, 3456 h cell,216구간 cell에 baseline/new MSE·signed delta·상호작용·보정 에너지·관측수·per-C 성분을 남긴다. ZERO의 c는 정확히0이고 prediction과 baseline bytes가 같은지 확인한다. 보정이 TUNE에서 줄인 오차를 DEV에서 늘렸다면 두 항 중 무엇이 달라졌는지 숫자로 답한다. 에너지 증가나 상호작용 부호 변화는 관찰된 산술 설명이며 분포 변화·과적합·학습량 부족의 유일한 인과 증명은 아니다.

F/FLAT·STATIC/RAMP·K는 각각 고정된 조건으로 모두 보존한다. 전역 평균의 개선이 특정 채널·h 악화를 숨기는지도 per-C/per-h로 보여준다. 입력 공간·checkpoint·선정 계수·원 예측 artifact SHA·origins·target/mask/scaler를 결속하고 원본을 덮지 않는다. 실제 데이터/NPZ 계산은 ROOT만 수행하며 SOURCE/작은 합성 검사는 위임 가능하다. 출력은 별도 신규 run/audit/한국어 연구 보고·그림으로 보존한다. TEST 접근·기존 중단 학습 재개·자동 commit/push·논문/PPT는 하지 않는다.
