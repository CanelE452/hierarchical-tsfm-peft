# Beijing K6: B2와 과거 창 지연 encoder의 최소 비교

**[확인] 아래 새3조건은 같은 CPU V43의 실제 전체 추론, 별도 FP64 재점수, freshCPU6 checkpoint replay 및 B1/B8 전체 물리입력 비용 증거가 갖춰진 범위다. 전체 MASTER·충분 학습·독립 일반화 완료를 뜻하지 않는다.**

[원정밀도 CSV](RESULTS_WITH_COST_v1.csv) · [입력·실제증거 SHA](REPORT_INPUT_BINDINGS_v1.json) · [봉인 계획](PLAN.json) · [실행기](beijing_minimum_runner.py)

B2를 기준으로 한 같은 K6/seed92601/CPU의 조건부 차이는 다음과 같다. 음수는 해당 오차 또는 latency가 작다는 뜻이며, 자동 모델 순위나 유의한 일반화 개선을 뜻하지 않는다.

```text
LAG_CAUSAL   MSE -6.17378%  MAE -2.67529%  B1 -0.542529%  B8 +0.457536%
LAG_BIDIR    MSE -12.5788%  MAE -5.527%  B1 -1.24641%  B8 +3.03708%
```

```text
role         selected-refit  search/refit-exposures search/refit-updates  B1 median/p90 s  B8 median/p90 s
B2           1.0 29880/12063 1868/754 0.0481541501/0.0505631502 0.32640055/0.34342452
LAG_CAUSAL   0.25 29880/3016 1868/189 0.0478929/0.0491748299 0.32789395/0.34051628
LAG_BIDIR    3.0 29880/36189 1868/2262 0.0475539502/0.0481637999 0.3363136/0.34515515
```

새3조건은 원 A50[0,10519), A_stop[10519,12622), fresh A60[0,12622), 반복 개발 V[14726,16830)의43 origins·20539 observed labels·C11/L512/H48를 유지했다. 검색0/.25/1/3에서 A_stop으로 epoch를 고른 뒤 새 모델을 그 길이만 refit했다. 최대 search 일정은 같아도 선택된 refit 길이·실제 exposures와 구조별 용량은 다를 수 있으므로 encoder만의 인과 효과나 같은 compute 비교로 확대하지 않는다. 원 channel macro 정규화 MSE를 주지표로 사용하며 MAE 차이도 함께 남긴다.

각 actual fit/replay/cost의 마지막 F tensor hash는 동결 Small의 원 hash와 같고 source48·선택 artifact/scaler·V 네 입력 지문이 연결됐다. 이는 구현·입력·동결 검사 통과이며 성능 개선이나 충분한 수렴의 증거가 아니다. causal encoder와 bidirectional encoder 모두 예측 origin 이전512-step 창만 사용한다. **bidir는 그 과거 창 안에서 뒤쪽 시점을 이용할 수 있으므로 시점별 과거 전용 표현이나 directed causal graph를 검증한 것으로 부르지 않는다.**

```text
role                 MSE          MAE          accuracy device / scope
B2                   0.8554261611 0.6092517182 cpu / NEW_CPU_V43_FP64_CPU6
LAG_CAUSAL           0.8026140272 0.592952462 cpu / NEW_CPU_V43_FP64_CPU6
LAG_BIDIR            0.7478236255 0.5755783651 cpu / NEW_CPU_V43_FP64_CPU6
RAW_SMALL            0.7968786537 0.5618622043 NOT_RECORDED_IN_MANIFEST / REFERENCE_DEVICE_NOT_RECORDED
G_INITIAL025         0.8091814806 0.5993137651 NOT_RECORDED_IN_MANIFEST / REFERENCE_DEVICE_NOT_RECORDED
G_DEEP_E3            0.8142803949 0.6066850921 NOT_RECORDED_IN_MANIFEST / REFERENCE_DEVICE_NOT_RECORDED
RIDGE_NATIVE         0.8315416368 0.6073577988 cpu_NATIVE_NUMPY_SOURCE_TRACED / CPU_NATIVE_NUMPY_SOURCE_TRACED
PCA_FIXED_NESTED_K6  0.779456596 0.5956357306 cpu / REUSED_CPU_V43_FP64_AND_CPU6
```

기존 Raw/G/Ridge/PCA 행은 exact ID의 같은 V/scaler/mask/target/origin/channel일 때만 인용했다. G 초기.25와 extended3-search는 별도 행이며 서로 대체하지 않는다. 원 Raw/G manifest에 장치가 명시되지 않으면 `NOT_RECORDED_IN_MANIFEST`와 reference scope로 남기고 CPU 동등 정확도 비교로 합치지 않았다. G의 training_device=cuda는 원 학습 recipe의 기록이며 그 자체로 전체 V의 실제 추론 장치를 증명하지 않는다. Ridge는 원 NumPy native 경로를 source-traced CPU로 따로 표시한다. 기존 nested PCA K6는 원 CPU V43 FP64/CPU6 및 N3 비용 감사의 재사용 행이며, 고정 basis inverse+center가 B2의 학습된 lag D/current anchor와 다르다. PCA 새 optimizer0회는 PCA 제조 비용0초를 뜻하지 않는다.

새 비용은 CPU2/row128, 각10warm·20timed·별도20intrusive stage이며 physical history/mask→과거 채움/정규화→P/E/F/D→FP64 inverse/CPU output을 매 trial 계산했다. cached Z/forecast는 쓰지 않았다. 같은 B1/B8 물리입력·mask SHA를 세 새조건에서 대사했다. 표의 median/p90은 실제 저장된 ordinary whole20회 요약이며 이 builder가 비용을 다시 측정하거나 통계 gzip을 해석한 것은 아니다. 단계 합이 whole과 같다고 하지 않는다. CPU resident weights·관측된 process-tree RSS·OS peak·GPU peak 및 학습/로드 비용은 별개의 범위다. 기존 Raw/G/Ridge의 정확한 CPU 비용 결속이 이 보고에 없으면 `NOT_INCLUDED`로 남겼으며 다른 장치 비용으로 채우지 않았다.

ROOT의 실제 fit큐 종료와 각 outer postflight/observed-tree return0·remaining[]·quiet 증거를 연결했다. 이 보고는 프로세스를 조회·종료하지 않으며 OS 전체 idleness나 미관측 후손 부재를 증명하지 않는다. 현재 numeric prefix0:16830과 inherited full Task SHA를 구분하고, 원 probe/full-loader의 역사와 반복 개발 V 관찰을 지우지 않는다. 새 protected T 수치/평가·새 모델/NPZ 해석·학습·비용 재실행은 보고 생성 과정에서0이다.

남는 판단은 이 반복 V/한 seed/제한된3epoch 검색에서 지연 encoder의 추가 가치와 비용이 어떻게 바뀌었는가이다. 대조학습·동적 graph 필요성, Bolt 유지의 일반적 우월성, 다른 자료/K/seed와 충분 학습·수렴은 별도 미확정이다. 후순위 B4/ADAPTS/LoRA family4와 기존 중단 shared/max 및 남은 MASTER 항목은 이3조건 완료에 포함하지 않는다.
