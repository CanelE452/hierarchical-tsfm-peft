# 교수님 공유용 요약 v5

현재 채택 가능한 결론은 다음과 같습니다.

- 실제 미래 정보 누출이 있었는가? 채택된 E1 forecast-boundary 검사 범위에서는 material future leakage가 보이지 않는다.
- 시점별 과거 전용 표현이 예측을 개선했는가? 현재 실제 FULL-vs-PREFIX paired case 8개에서는 PREFIX가 physical MSE를 낮춘 경우 5개, 높인 경우 3개이고, normalized MSE는 낮춘 경우 7개, 높인 경우 1개다.
- 다른 채널의 지연 정보가 실제로 유용했는가? 혼재다. Jena real VAL에서 cross-now가 ALL-lag보다 강한 구간이 있어 delayed increment 자체는 아직 입증되지 않았다.
- 표현·잠재 미래 예측·디코더 중 어디에 문제가 남았는가? E4 case별로 latent MSE 순위와 decoded raw MSE 순위가 달라질 수 있어 F 정확도만으로 설명하면 부족하다.
- 대조학습과 동적 그래프가 필요한가, 단순 대안으로 충분한가? E5/E6는 dataset·seed별 원수치로 판단해야 한다. Robin에서는 contrast가 REC보다 낫지만 raw SELF_LAG baseline보다 높고, Jena E6 UPDATED는 seed별로 SELF 대비 방향이 갈려 동적 그래프 이득이 안정적이지 않다.
- Chronos-Bolt를 유지할 근거가 있는가? 채택된 실제 case 기준으로 PREFIX/FULL은 physical 5/8, normalized 7/8 개선이고, E4 Bolt latent는 SELF 대비 physical 6/8, JOINT 대비 physical 6/8 개선이다. 따라서 유지/폐기는 전체 조건부 원수치로 판단해야 하며, 단일 Jena d16 결과로 폐기하면 안 된다.

핵심 제한은 TEST 미사용, 반복 VAL 진단, heterogeneous channel units, final proof PASS 항목만 채택한 범위입니다: E3/E4 adopted cases=8, E5 adopted runs=2, E6 adopted runs=3, Bolt adopted runs=1, Cost closure adopted runs=5. REF latent는 분리 진단일 뿐 deployable bound가 아닙니다.
검증 proof status=PASS, excluded=0.

E6 그래프 결과가 포함되어 있으므로 mode별 비교는 본 보고서의 E6 원수치 표를 기준으로 해석하면 됩니다.
