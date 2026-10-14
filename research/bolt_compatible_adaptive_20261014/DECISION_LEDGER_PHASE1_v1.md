# 결정 기록: 승인된 Phase 1

## D1 — 같은 시간 특징에서 다른 채널이 유용한가?

1. **질문:** SELF33과 JOINT33의 관측 원미래 TUNE 오차를 동일 학습·α 선택 조건에서 비교하면 채널 정보의 실용 이득이 남는가?
2. **현 증거:** [확인] 기존 비교는 자기 채널·joint의 특징 또는 α 선택 조건이 달랐다. 새 runtime9 및 readout8 수치 fixtures PASS. 현재 실제 새 학습 결과는0이다. [추정] 일부 joint 악화는 공통 정보 부족뿐 아니라 차원/작은 FIT/과적합 때문일 수 있다.
3. **경쟁 설명:** (a) 다른 채널의 조건부 예측 이득이 작음 (b) 정보가 있으나 고차원/규제가 불리함 (c) 시간 요약·선택 조건이 결과를 흐림. ridge의 표현 능력 자체도 한계다.
4. **다음 검사:** Robin17·Beijing11, FIT256/TUNE64/H48/L512, 동일 FP32 actual-Y−last target 및 FIT 특징 통계로 SELF33/JOINT33/SELF512. α7+PERSIST+sharedMEAN의9pool을 전체 macro TUNE MSE로 선택. 원 JOINT14후보는 source/수치 gate 후 재사용; 실패 경로만 재학습. 신경망2seed를 deterministic fit에 붙이지 않는다. TUNE MSE≥1% 개선·MAE 악화≤1%·4block 중3개 이상 개선을 탐색 분기 gate로 고정한다.
5. **실행 결과:** 진행 중. 이후 실제 FIT/TUNE 원수치·rank/계수/표본·추론·자원 receipt·검산을 기록한다.
6. **다음 결정:** FIT/TUNE 기준으로 먼저 분기를 잠근 후 개발 지표를 계산한다. gate가 부정이고 FIT만 개선되면 같은 FIT 시간 구간에서 최대512 legal origins의 self/joint1회 추가 진단을 허용한다. gate가 긍정이면 최대2표현 계열·선택된 native 대조·동등 RAW33 보정을 필요에 따라 진행한다. Raw/SELF512/Raw+동등 보정 대비 정확도·whole 비용은 별도 채택 판단이다.
