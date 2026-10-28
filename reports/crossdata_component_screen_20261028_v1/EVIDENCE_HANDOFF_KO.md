# 실험 근거 전달

발표 파일·대본은 생성하지 않았다. 아래는 작성 담당자가 사용할 수 있는 실험 범위와 제한이다.

# 교차 데이터 구성요소 탐색 결과

PeMS32·Beijing은 과거 개발 V 결과를 참고해 선택했다. 새 미관측 데이터 일반화 시험이 아니다. K는 PeMS32=8, Beijing=6으로 고정했으므로 K 또는 Kτ 선택 규칙을 검증하지 않는다.

각 데이터의 fresh OURS와 같은 seed·전체 4구성이 모두 끝난 base만 비교한다. 선택 seed92603은 네 구성 모두 완료된 경우에만 함께 집계했다. 부분 결과는 ALL_NEW_RUNS/JOB_STATUS에 보존한다.

- pems32: base 8/8, optional 0/4, 비교 n=2 (PILOT).
- beijing: base 1/8, optional 0/4, 비교 n=0 (INCOMPLETE).

차이(+)는 제거/고정한 구성의 오차가 더 커 fresh OURS가 낮다는 뜻이다. 중앙값 차이와 seed별 차이의 중앙값은 다른 통계이며 표에서 분리했다. 중앙값 차이가 두 IQR의 최댓값을 넘고 같은 방향 seed가 ceil(2n/3) 이상이면 기술적 방향만 표시한다. 유의성·동등성 검정이 아니다.

- pems32 / TIME_ONLY: MSE 중앙값 차이 -1.17%; 기술적 방향 ARM_LOWER; 상태 PROVISIONAL.
- pems32 / A5_LAG0: MSE 중앙값 차이 +1.50%; 기술적 방향 FULL_LOWER; 상태 PROVISIONAL.
- pems32 / E_FIXED: MSE 중앙값 차이 +0.13%; 기술적 방향 UNCLEAR; 상태 PROVISIONAL.

## 해석과 한계

- n=2는 전부 PILOT/잠정이며 n=3도 탐색 결과다. 어느 한 구성이라도 cap 선택수가 ceil(2n/3) 이상이면 해당 비교는 잠정이다. 좋은 방향이더라도 충분한 최적화나 확정적 구성요소 효과를 주장하지 않는다.
- PeMS32의 cap6은 모든 구성에 동일하며 추가 연장하지 않았다. Beijing은 모두 cap3이다. 같은 cap이 같은 실제 선택 epoch나 계산량을 뜻하지 않는다.
- TIME_ONLY는 채널 MLP를 제거하지만 공동 LayerNorm은 유지한다. 따라서 채널 상호작용 전체의 제거, 시간축 자체의 보편적 우월성 또는 용량을 완전히 통제한 대조가 아니다.
- A5_LAG0는 직접 anchor와 lag0를 유지한다. 추가 lag 참조의 구성 효과이며 개별 lag의 인과성·최적성이나 모든 관측값의 필요성 시험이 아니다.
- E_FIXED는 search/refit 각각의 PCA 초기 E를 고정한다. E 학습 여부의 대조이며 PCA 초기화, 압축의 최적성, K 규칙 또는 독립 테스트 우월성을 단독으로 입증하지 않는다.
- 다른 데이터의 과거 좋은 결과와 새 결과를 하나의 seed 표본으로 섞지 않는다. 미완료·실패·예산 제외도 JOB_STATUS에 모두 공개한다.
