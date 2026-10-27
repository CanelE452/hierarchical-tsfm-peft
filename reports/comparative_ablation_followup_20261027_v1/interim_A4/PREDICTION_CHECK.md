# Fixed planner prediction check

[확인] 아래 기준은 후속 지시문 §5의 사전 고정 예측이며 등록 판정을 바꾸지 않는다.
A4 effect(%) = 100 × (A4 median − OURS median) / OURS median.
ETTh1·ETTh2: effect ≥ +50%. Jena·Appliances: |effect| < 10% (경계 10%는 불일치).
D1: ETTh1·ETTh2에서 A4·C2·C3 각각의 **job별 하한/같은 대상점 실제 MSE 비율의 중앙값** ≥ 0.5.
D1 집계는 두 중앙값의 비율이 아니다. 원문은 집계 방식을 명시하지 않아 데이터별 중앙값 보고 규칙에 맞췄다.
n<10 또는 정해진 seed가 빠지면 INCOMPLETE이며 예측의 성공·실패를 주장하지 않는다.

```text
kind       dataset              method n/10 value           prediction status
A4_EFFECT  jena                 A4     10   -1.747846       MATCH
A4_EFFECT  ETTh1                A4     10   155.949048      MATCH
A4_EFFECT  ETTh2_safe17400      A4     10   225.337055      MATCH
A4_EFFECT  appliances           A4     10   18.571226       NOT_MATCH
D1_RATIO   ETTh1                A4     10   0.644605        MATCH
D1_RATIO   ETTh1                C2     10   0.627389        MATCH
D1_RATIO   ETTh1                C3     10   0.629555        MATCH
D1_RATIO   ETTh2_safe17400      A4     10   0.767176        MATCH
D1_RATIO   ETTh2_safe17400      C2     10   0.708615        MATCH
D1_RATIO   ETTh2_safe17400      C3     10   0.692266        MATCH
```

[추정] D1의 큰 비율은 출력 공간 제약과 수준 이동 설명에 부합할 수 있으나 마지막 관측값·lag의 개별 인과 효과를 식별하지 않는다.
