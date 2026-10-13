# 평가 기준을 맞춘 학습·압축·예측기 진단

2026-10-13. 사용자가 선택한 **FIT 통계 기반 정규화 channel-macro MSE**에 학습 손실, 미래 TUNE checkpoint 선택과 보정 lambda 선택을 맞춘 후속 실험이다. Jena·Robin의 기존 TRAIN/VAL을 사용했다. 반복 사용한 개발 VAL 결과이며 독립 일반화 검증이 아니다.

## 결과와 판단

[확인] 낮은 학습률과 weight decay는 작은 K에서의 성능 격차를 해결하지 못했다. REC 12개 조건 모두 FIT 복원 오차는 줄었지만, FIT 이후 TUNE 위치의 복원 오차는 커졌다. 따라서 “복원은 성공했는데 예측만 실패했다”는 전제를 둘 수 없다.

[확인] 성분 수 K를 늘리면 두 자료의 주오차가 크게 줄었다. Jena의 예측 목적 학습에서 K=5의 두 seed 평균 VAL 정규화 MSE는 0.993492, K=10은 0.879588, K=21 항등 초기화는 0.776536이었다. 원자료 Bolt는 0.848632다. **K=21 결과는 압축의 이득이 아니라, 압축 없이 E/D를 예측 목적으로 학습한 입력·출력 적응의 근거다.** 원단위 MAE는 원자료 Bolt보다 나빠 모든 지표의 개선이라고 부르지 않는다.

[확인] Robin의 K=4 평균은 0.599845, K=8은 0.482768, K=17 전체 PCA는 0.326012였다. 원자료 Bolt는 0.340402다. 전체 PCA의 두 seed는 모두 학습 전 checkpoint를 선택해 기능적으로 같은 예측을 한다. 이 개선은 고정 회전의 결과이며 표현학습의 개선이나 독립된 두 번의 일반화 성공이 아니다. 원단위 MAE도 7.102469로 원자료 Bolt의 5.938975보다 나빴다.

[확인] 압축의 비용 교환도 남는다. Jena K=10은 원자료 Bolt보다 주오차가 약3.65% 크지만, 단독 재측정 batch8의 창당 중앙 시간은 평균2.400ms로 원자료3.835ms보다 약37.4% 작았다. batch1에서는12.741ms 대12.024ms로 이 절감이 나타나지 않았다. 이를 정확도 개선이나 모든 배포 조건의 속도 우위로 해석하지 않는다.

[판단] 현재 근거는 예측 목적 학습의 가능성과 작은 K에서의 손실을 분리하라는 쪽이다. 다음 연구 질문은 **“과거 입력을 변환해 동결 TSFM에 연결할 때, 어느 압축률과 실제 추론 비용에서 원자료 기준군보다 유용한 결과를 얻는가?”**로 좁히는 것이 적절하다. 작은 K의 정확도 우위, 대조학습이나 동적 그래프의 필요성은 아직 입증하지 못했다. 이번에는 이 후속 방법들을 추가 실행하지 않았다.

## 읽을 자료

- [한국어 상세 보고서](report_v1/REPORT_KO.md): 질문별 원수치, 두 seed, 반대 방향 지표, 비교 조건과 한계.
- [전체 방법 원수치](report_v1/all_methods.csv), [실제 학습 조건](report_v1/unique_training.csv), [단독 비용 재측정](report_v1/batch_cost.csv).
- [실행 정본](run_resolved_v1/RESULT.json): 최초 성공 결과와 별도 수리 결과를 연결하고 원래 비용을 보존한 manifest.
- [독립 재채점·선택 검산](validation_resolved_v1/audit.md), [원자료·FIT 통계·정답·마스크·입력 경계 검산](validation_source_v1/audit.md), [실행 계획](PLAN_KO.md), [시작 자산 검사](PREFLIGHT.json).

![성분 수와 시간 범위의 영향](report_v1/g2_components_history.png)

![정확도와 실제 배포 비용](report_v1/g3_accuracy_latency.png)

## 실행 상태와 보존

[확인] G0 재사용·새 추론, G1 24개 및 G2 20개 유효 핵심 학습 조건, G3 동일 E/D의 예측기·복원·원자료 기준군을 실행했다. 정본은 70개 평가 case·236개 방법·472개 batch 비용 기록이다. 실제 수리는 중복 시도를 포함하므로 핵심 학습 48회 시도 중 45회가 600 update를 완료했고, native는 33회가 600 update를 완료했다. 유효한 서로 다른 native 조건은 32개다.

Windows 결과 기록 잠금 2경로와 Jena 상수 rain 채널의 비유한 gradient 2경로는 [최초 실행](run_full_v1/RESULT.json)에 실패 기록으로 남겼다. [수리 실행](run_repair_v1/RESULT.json)은 해당 네 조건을 새로 학습했다. 분산0에서 backward0을 사용하는 수치 convention은 CPU/CUDA 검사에서 forward를 보존했으며, 그 범위와 학습 결과를 보고서에 명시했다. 원래 실행 소스와 결과는 덮지 않았다.

초기 비용 측정과 별도 보충 학습이 겹쳤으므로, 모든 Python GPU 학습 종료 후 [236개 방법의 비용을 다시 측정](run_clean_cost_v1/RESULT.json)했다. 추가 학습은 없다. 보충 실험의 실패·완료 시도까지 포함한 GPU 사용 process wall 합은 2065.271초(약34.42분)로 2700초 상한 안이다. 동시에 실행된 시간도 합산한 값이며 실제 elapsed wall이나 순수 GPU kernel 시간이 아니다.

[별도 평가 기준 정렬 보충 결과](../forecast_metric_alignment_20261013/README.md)는 본 정본의 236행에 섞지 않았다. 비용에는 포함했다. 기존 연구·결과·다른 프로젝트의 작업은 보존했고 보호 TEST를 읽지 않았다. 이번 후속은 commit/push하지 않았다.

## 코드와 로컬 자산

[experiment.py](experiment.py)는 G0–G3, [repair_experiment.py](repair_experiment.py)는 이번 최초 실행의 실패 경로, [measure_resolved_cost.py](measure_resolved_cost.py)는 저장된 모델의 비용, [resolve_results.py](resolve_results.py)는 정본 병합을 담당한다. [make_report.py](make_report.py)는 원수치에서 CSV·PNG·SVG를 생성한다. 실행별 source snapshot과 SHA가 해당 결과 폴더에 있다.

TRAIN/VAL 원자료, 기존 G0 checkpoint, pinned Chronos-Bolt-small과 신규 예측 NPZ·모델 state는 로컬 `.cache` 자산이다. Markdown·그림만으로 학습을 재현할 수 있다고 주장하지 않는다. 검산에 사용한 실제 파일 경로와 SHA는 실행 정본과 검증 기록에 있다. 기존 resource ledger를 지우거나 이전 캠페인을 자동 재개하는 재실행은 하지 않는다.
