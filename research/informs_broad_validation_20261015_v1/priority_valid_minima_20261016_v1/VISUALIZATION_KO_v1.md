# N4 Beijing K6 시각자료

![B2와 지연 encoder의 정확도 및 실제 전체 비용](figures/BEIJING_MINIMA_v1.svg)

[원정밀도 CSV](RESULTS_WITH_COST_v1.csv) · [실험 보고](RESULTS_KO_v1.md) · [독립 감사](INDEPENDENT_ARITHMETIC_REVIEW_ROOT_v1.json) · [그림 결속](FIGURE_BINDINGS_v1.json) · [코드](plot_beijing.py)

새 B2/causal/bidir 세 조건만 같은 순서로 표시했다. 모든 축은0부터이며 표시 숫자만 반올림하고 원 CSV와 실제 artist 값은 바꾸지 않았다. 지연시간은 CPU2 ordinary whole20 median의 **초/batch**이고 B8은8개 batch 전체다. 확정된 p90만 위쪽에 표시하며 신뢰구간이나 seed 분산이 아니다.

같은 K6 · frozen F · forecast MSE 학습 · 구조별 parameter capacity와 선택 refit 길이는 다를 수 있음. 한 seed · 반복 개발 V · 제한 학습: 수렴·독립 일반화 미확정. Bidir는 과거 창 내부 양방향이며 시점별 causal 검증이 아님.

PCA6 점선은 exact N3 CPU2 동일 V43/A60의 기존 MSE·MAE 참고다. 고정 basis transpose+center와 학습 lag D/current anchor가 달라 동일 decoder 비교가 아니다. PCA 비용은 이 그림에 섞지 않았다.

Raw/G/Ridge의 장치·비용이 다른 기존 행은 위 실험 보고 링크로만 제공한다. 자동 모델 순위, 수렴·독립 일반화 또는 방법론 성공을 주장하지 않는다. 새 통계·trial 재집계, 원자료/Task/NPZ/CP/model 해석, 학습·추론·비용 재측정·프로세스 제어·Git은0이다. SVG는 게시용, PNG는 private preview다.
