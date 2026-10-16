# 고정 PCA 6조건 그림

![동일 자료의 Raw 대비 PCA 오차와 PCA CPU 지연시간](PCA_ACCURACY_CPU_COST.svg)

입력은 저장 집계 CSV 6행뿐이다. 실제 새 추론·학습·비용 계측은 하지 않았다. MSE/MAE는 자료별 동일 V/data/scaler/cohort에 한정하고, 자료 간 평균은 만들지 않았다. Raw 4개 기준선을 K별로 재표시한 것이므로 독립 기준선 6개가 아니다. PCA의 optimizer update는 0이며 CPU 비용은 2 threads의 timed20 전체경로 median/p90다. p90는 신뢰구간이 아니며 stage_probe 시간은 섞지 않았다. 같은 CPU Raw 비용이 없어 속도 배율·Pareto 우위는 주장하지 않는다. T 관측 후 개발·단일 seed·독립 일반화 미검증 범위를 유지한다.

입력 원바이트는 [INPUT_AGGREGATE.csv](INPUT_AGGREGATE.csv), 생성·출력 SHA는 [FIGURE_BINDING.json](FIGURE_BINDING.json)에 있다. PNG는 300 dpi로 함께 생성한 로컬 출력이며 게시 형식 선택은 별도다.

이 결과 폴더에서 새 출력 폴더로 재생성할 수 있다:

```text
python ../../../render_fixed_pca_summary.py --input INPUT_AGGREGATE.csv --input-sha 8e001bbb8b1ec5d1b3622800de2b30d87e9f3a3cc523259028f347e73c1facda --output ../RENDER_REPLAY_NEW --execute
```
