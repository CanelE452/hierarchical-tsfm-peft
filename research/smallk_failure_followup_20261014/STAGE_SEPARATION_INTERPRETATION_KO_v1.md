# 이번 분리 진단에서 확인한 것과 남은 질문

[실제 보고서와 그림 6개](report_stage_separation_v2/REPORT_KO.md) · [112 후보·48 선택 예측 검산](validation_v1/STAGE_NATIVE_AUDIT_v4.json) · [36 경로 비용 검산](validation_v1/STAGE_NATIVE_COST_AUDIT_v1.json) · [파일·링크·수치표 보존 검증](validation_v1/STAGE_REPORT_FINAL_VERIFICATION_v2.json)

## 결론

[확인] 현재 고정 E의 33K 과거 특징에서 학습한 ridge와 디코더 보정은 정확도를 해결하지 못했다. 과거 표현에 정보가 전혀 없다는 결론도 맞지 않는다. N2는 같은 current-C anchor를 쓰는 persistence와 FIT 평균 보정보다 좋다. 그러나 두 데이터·두 seed 모두 원 RAW와 MAIN보다 나쁘다. 재구성 목적 때문이라는 설명도 현재 모델에는 그대로 적용되지 않는다. 원 E/D는 frozen F를 통한 실제-Y 예측 손실로 학습했다.

```text
두 seed 평균 정규화 MSE        Robin       Beijing
RAW                           0.340402    0.589575
MAIN                          0.569547    0.661957
N1: 과거 E → 미래 latent → D  0.914009    0.727793
N2: 과거 E → 미래 C           0.924093    0.736626
N3: 고정 N1 + D 보정          0.946790    0.737978
Persistence                   1.308464    0.908442
FIT 평균 보정                 1.305133    0.901634
O0: 실제 미래 E + 원 D        0.427725    0.411924
O1: 실제 미래 E + D 보정      0.455678    0.380217
```

## 어디가 문제인가

[판단] Beijing은 실제 미래 표현을 주면 낮은 오차가 가능하지만 과거에서 그 표현을 예측하는 새 경로는 실패했다. 미래 표현의 조건부 정보는 남아 있으나, 과거 예측 가능성·teacher 목표·feature 요약·선형 예측기 제약을 나누어야 한다. O0/O1은 실제 미래를 사용하므로 배포 모델이나 이론적 upper bound가 아니다.

Robin은 실제 미래를 사용하는 O0/O1도 RAW보다 나쁘다. 기존 E·현재 anchor·D 함수에서 손실이 남는 조건이라는 증거다. 임의의 encoder/decoder로도 불가능하다는 증거는 아니다. O1도 O0를 개선하지 않았으므로 outgoing decoder 보정 하나로 해결된다고 주장할 수 없다.

N3는 선택된 N1을 고정한 비교다. 양 데이터·양 seed 모두 VAL/EVAL이 나빠졌다. 일부 FIT 오차가 줄어도 평가 개선으로 이어지지 않았다. 새 decoder 학습 자체의 미실행이나 단순 학습 횟수 부족으로 설명할 근거가 없다. 현재 선형 ridge 계수는 실제로 추정했지만, 검산은 SVD/lstsq 최적성을 독립적으로 다시 푼 검증은 아니다.

Robin teacher는 모든 horizon·K에서 FIT 256개이고 미관측 출력은 0개다. 따라서 teacher 결측을 Robin 실패의 원인으로 사용할 수 없다. Beijing은 complete-prefix teacher가 1-step 225개에서 48-step 83개로 감소한다. 장기 teacher는 완전 관측 구간에 조건부이며, N1과 N2는 감독 표본과 목표 좌표가 다르다. 두 차이를 하나의 원인으로 합치면 안 된다.

## 압축·지연·대조학습과 Chronos에 대한 판단

[확인] 새 N2는 persistence/평균 보정보다 좋지만, 같은 direct-C readout·anchor·학습 규칙의 원 C 과거 입력 대조가 없다. 기존 RAW_SELF33/RAW_JOINT_RIDGE는 재사용했고 원 배열 조건을 검증했지만 readout 구조와 anchor가 같지 않다. 따라서 E 자체의 필요성이나 압축의 이점은 미확정이다. 512-point latent를 32개 block mean과 마지막 값으로 줄이는 33K 요약이 지연 정보를 잃는지도 이번 비교에서 분리하지 않았다.

[판단] 다음 우선 질문은 같은 actual-Y direct-C 예측기에 원 C의 33-block 특징과 E의 전체 512K 과거를 넣었을 때 어디서 차이가 생기는지다. 이 단계에서는 대조학습·동적 그래프의 필요성이나 지연 정보의 인과적 유용성을 확정할 수 없다. 큰 구조를 먼저 추가하는 근거로 이번 실패를 사용하면 안 된다.

Chronos 경로를 현재 기준선으로 유지할 근거는 있다. 같은 원 E/D에서 MAIN이 새 N1/N2/N3보다 낫고 RAW가 더 낫다. 이는 Chronos의 보편적 필요성이나 모든 native 모델의 열세를 입증하지 않는다.

## 비용과 적용 범위

[확인] 현재 hybrid 구현의 B1 전체 처리 시간은 native 2.66–3.93ms, RAW 48.30–51.64ms였다. 전송과 CPU ridge/D 계산을 포함한 같은 실행의 측정이다. 정확도는 동시에 나빠졌으므로 정확도·비용 목표를 달성하지 못했다. 비용은 첫 1/8개 노출 창을 반복한 측정이고, 이전 캠페인보다 RAW 절대 시간이 커져 캠페인 간 속도 개선률로 비교하면 안 된다. method별 독점 메모리나 최적 구현의 비용도 측정하지 않았다.

학습 FIT256, 선택 TUNE64, L512/H48, Robin 반복 VAL30·Beijing 노출 EVAL146, 원 FIT 통계와 mask, 원 MAIN seeds92601/92602를 유지했다. 새 TEST·독립 holdout을 쓰지 않았다. prefix·교란 검사는 제한된 실제 입력에서 통과했고, 보편적 누수 부재나 독립 일반화를 증명하지 않는다.

## 검산과 보존

Native 최종 검산 5,420/0, 비용 검산 3,448/0, 최종 파일·링크 검사 78/0이다. 최초 NPZ hash schema 해석 오류와 Torch/erf GELU 차이로 실패한 검산 버전은 보존했고 원 학습 결과를 바꾸지 않았다. 보고서 v1의 그림 배치를 고친 v2에서 CSV 7개는 byte SHA가 같다. 그림 6개를 직접 확인했다. 시각 자료 2번은 위쪽 Robin/아래쪽 Beijing, 왼쪽 MSE/오른쪽 MAE이며 O0/O1은 미래 진단이다.

현재 HEAD는 95585946f99da9b50cbdadcadd7441ccbf6ff2e5다. 이번 범위에서 commit/push를 하지 않았다. 이번 진단은 완료됐지만 전체 연구의 정확도 문제는 해결되지 않았다.
