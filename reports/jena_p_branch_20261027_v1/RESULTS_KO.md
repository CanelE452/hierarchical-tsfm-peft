# Jena P 분기 결과

개발 검증 V의 탐색 결과다. 기존 P 기여를 본 뒤 Jena를 선택했으며 보호 T는 평가하지 않았다.

기본 실행 20/20개, 추가 연장 0개 완료. 연장 10개 전체가 끝난 경우에만 해당 방법의 주 결과로 사용했다.

```text
Method                    n      median MSE      IQR       parameters
No P                         10     0.775362    0.017211      1008
Time branch only             10     0.698929    0.024693     34320
Channel branch only          10     0.760005    0.019675      2405
Full P                       10     0.697770    0.021174     35717
A2: large channel replacement 10     0.723501    0.028942     35708
```

- Time branch only: Full P 대비 MSE +0.17%, 기술적 판정 UNCLEAR; seed별 차이 중앙값 -0.011873.
- Channel branch only: Full P 대비 MSE +8.92%, 기술적 판정 FULL_P_LOWER; seed별 차이 중앙값 +0.060444.
- No P: Full P 대비 MSE +11.12%, 기술적 판정 FULL_P_LOWER; seed별 차이 중앙값 +0.077592.
- A2: large channel replacement: Full P 대비 MSE +3.69%, 기술적 판정 UNCLEAR; seed별 차이 중앙값 +0.038675.

양수는 해당 변형의 오차가 더 크다는 뜻이다. 원 중앙값/max-IQR 규칙은 유의성 또는 동등성 검정이 아니다.

해석 범위: 현재 구현에서 분기를 제거하는 효과다. 시간 분기와 채널 분기의 용량이 다르고 joint LayerNorm/E/D 결합이 남으므로 시간축 자체의 우월성, 완전한 채널 독립성, 다른 데이터에서의 효과를 주장하지 않는다.

기존 C2/C3 비교의 잠정 상태는 이번 실험으로 해소되지 않는다. No-P는 기존 cap10 연장, Full-P/A2는 기존 cap3 결과이며, 새 변형의 일정과 상한 상태는 SUMMARY.csv와 EXECUTION_COMPLETION.json에 공개한다.

그림: FIG_JENA_P_BRANCHES.png / .svg / .pdf. 모든 seed 점을 표시했다.
