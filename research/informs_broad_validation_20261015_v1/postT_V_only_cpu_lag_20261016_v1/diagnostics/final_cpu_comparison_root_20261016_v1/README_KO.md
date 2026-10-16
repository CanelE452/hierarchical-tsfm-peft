# Jena 최종 CPU 비교 집계 그림

ROOT가 별도로 검산한 집계 CSV의 정확한 bytes만 사용했다. 렌더러는 Task·NPZ·가중치·개별 정답/오차 로그·원 비용 자료를 열지 않으며 새 검산이나 선택을 수행하지 않는다.

- [정확한 원 CSV 복사](INPUT_SNAPSHOT.csv)
- [run ID·입력·그림 SHA와 적용 범위](FIGURE_BINDING.json)
- [정확도 SVG](ACCURACY_MSE_MAE.svg) / [PNG](ACCURACY_MSE_MAE.png)
- [CPU B1/B8 비용 SVG](CPU_WHOLE_COST_B1_B8.svg) / [PNG](CPU_WHOLE_COST_B1_B8.png)
- [정확도-비용 SVG](ACCURACY_CPU_COST_SCATTER.svg) / [PNG](ACCURACY_CPU_COST_SCATTER.png)

정확도는 저장된 전체 V43 정규화 channel-macro MSE/MAE다. CPU의 6 origin checkpoint 재추론과 구분한다. A_stop 선택 epoch, 검색3epoch, 선택한 fresh A60 refit 길이는 같은 개념이 아니다. Raw·ridge에는 신경망 epoch를 붙이지 않았다.

FAILED/NOT_MEASURED 비용은 0이나 다른 조건의 시간으로 대체하지 않았다. 비용 선은 timed20의 min–max, 원은 median, 삼각형은 p90이며 CI가 아니다. CPU2·warm10/timed20/separate-stage20·row128·checkpoint off의 fresh whole physical pipeline 조건만 허용한다. 속도 배율은 계산하지 않았다.

T 관측 이후의 탐색적 V 재사용 비교, seed 하나다. 서로 다른 구조·학습량의 비교이며 동일 파라미터/연산량·충분 수렴·독립 일반화나 원 연구 전체 완료를 뜻하지 않는다. G는 CSV의 명시적 A-only CPU fresh ID만 허용한다.
