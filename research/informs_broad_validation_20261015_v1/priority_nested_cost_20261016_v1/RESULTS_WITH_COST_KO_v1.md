# Beijing 고정 nested PCA: 기존 V와 새 CPU 비용6개

**[확인] 같은 고정 A60 전체기저의 K3/6/11 기존 V를 재사용하고, 각 K의 B1/B8 물리입력 전체 추론 비용6개를 실제 측정했다. 신규 학습·정확도 재추론은0이며 전체 MASTER 완료가 아니다.** 정확도는 같은 반복 개발 V43/seed92601의 조건부 결과다.

[원정밀도 CSV](RESULTS_WITH_COST_v1.csv) · [독립 비용 감사](INDEPENDENT_COST_REVIEW_ROOT_v1.json) · [실행 계획](PLAN.json) · [원 계획](../postT_V_only_cpu_multik_20261016_v1/PLAN.json) · [입력 결속](REPORT_INPUT_BINDINGS_v1.json)

```text
 K MSE         MAE         B1 median s  B1 p90 s   B8 median s  B8 p90 s
 3 0.797925037 0.613960691 0.0305150999 0.0315190101 0.15942465 0.16381007
 6 0.779456596 0.595635731 0.0464319501 0.0469850597 0.32212945 0.33366889
11 0.75178996 0.576547512 0.0775629 0.08006237 0.57407645 0.58300091
```

Beijing C11/L512/H48, A50end10519/A60end12622, V[14726,16830)의43원점·20,539관측 정답과 같은 scaler3348bcf1…93fb를 유지했다. 오차는 원 C 채널의 A 정규화 척도다. 기존 전체 V FP64 재점수와 CPU6(4+2) checkpoint 재추론은 정확한 원 영수증으로 재사용했다. 이 보고 단계에서는 새 모델·NPZ 숫자·원자료를 읽거나 비용을 다시 측정하지 않았다.

공통 artifact의 A60 PCA basis11×11과 center를 그대로 쓰며 K는 같은 basis의 앞3/6/11축이다. K11도 회전된 좌표에서 F를 거친 뒤 basis transpose로 복원하므로 Raw 항등 경로와 같지 않다. 학습된 B2 lag D·현재값 anchor와도 복원 구조가 다르다. K 증가의 V 차이를 학습된 시간·채널 압축의 우월성으로 해석하지 않는다.

```text
 K B1 CPU weights MiB  B8 CPU weights MiB  B1 model load s  B8 model load s
 3 182.030323 182.030323 4.5507213 4.507318
 6 182.030323 182.030323 4.1839173 4.4522767
11 182.030323 182.030323 4.1838701 4.211372
```

비용은 CPU2/row128, 각10warm·20timed·별도20stage로 총300trial이다. 원 physical history/mask→과거 채움·정규화→고정 projection→동결 Bolt→basis inverse→FP64 역변환·CPU 반환을 측정했고 cached Z/forecast는 쓰지 않았다. 논리·물리 stream은 각 trial의 B×K, B8K11도88stream이어서 모두 실제 F call1회다. stage는 intrusive 별도 범위며 stage 합이 whole과 같다고 하지 않는다. 모든 실제비용 종료·quiet·source48·수치 재집계는 독립 감사의 범위로 연결한다.

모델 적재와 resident weight MiB는 steady-state latency 및 sampled peak RSS와 별개다. K를 줄여도 동결 backbone resident weights가 같은 점을 CSV에서 확인할 수 있다. 표의 적재 시간은 cold disk cache나 전체 Python/data startup 비용을 뜻하지 않는다. PCA 기저 제조 시간은 원 영수증에 별도 기록이 없어 `NOT_RECORDED`다. 새 optimizer0회는 PCA fit 비용0초를 뜻하지 않는다.

이번 숫자 변환은 원 source의0:16830과 작은 channel metadata로 제한됐으며 새 보호 T 수치·target·평가는0이다. 전체 압축파일의 opaque SHA/ZIP I/O와 numeric parse 경계를 구분하고, 옛 CPU6 loader의 역사적 물리 역직렬화 이력을 지우지 않는다. 고정 transform은 신경망 수렴의 판정 대상이 아니며, 이 결과가 원 신경망 학습의 수렴 미확정을 해결하지 않는다.

결론은 한 자료·한 seed에서 고정기저의 비용/오차가 어떻게 달라지는지 확인했다는 것이다. 무조건적인 전체 정확도·메모리·총학습비용 우위나 독립 일반화 근거가 아니다. 다음 기존 Beijing B2/causal/bidir K6 최소3조건은 아직 준비이며 원예산·보호 T 계약 안에서 별도 실행한다. 중단 shared/max·미시작13/core29·M5·다른 자료/K/학습량 요구는 여전히 별도 잔여다.
