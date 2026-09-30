# v11 판단: 내부 LoRA는 조건부 유지, 공간 학습은 비교 근거로 유지

**[확인] 이번 비교에서 정확도 손해가 전반적으로 해결되지는 않았다. 그러나 Jena에서 A의 내부 적응 이득과 정확도–메모리–처리량 절충이 함께 남았다. A를 좁은 후속 연구 후보로 유지하고, B의 공간 학습은 별도의 유효한 내부 대조 결과로 유지한다.** 세 자료를 아우르는 최고 정확도·보편적 속도 우위·독립 확증·논문 최초성을 확보했다는 판정은 아니다.

근거는 [고정 평가](evaluation01.json), [모든 기간·seed 값](evaluation01_models.csv), [채널별 값](evaluation01_channels.csv), [GPU/CPU 비용](cost_summary.json), [결과 표·출처](report_values.json)다. [최종 숫자 검산](final_checks.json)이 통과했으며 자체 검산을 독립 재현으로 부르지 않는다. 게시 상태는 [STATUS.md](STATUS.md)와 게시 영수증을 따른다.

## 외부 대안 앞에서의 위치

아래는 TRAIN 표준화 좌표에서 TEST-A/B의 관측 오차합과 관측수를 채널별로 합친 뒤 채널 평균, 마지막에 두 seed **손실**을 평균한 값이다. 예측 ensemble이 아니고 자료 간 원 MSE를 합산하지 않는다. Robin·Jena·Hog는 모두 이미 노출된 개발 자료다.

```text
Dataset  Model                MSE        MAE
Robin    A                    0.273071   0.323059
         B learned U          0.288584   0.328083
         B learned + Gamma    0.292107   0.331575
         F0                   0.229095   0.262074
         LoRA native          0.223610   0.256347
         LoRA full MSE        0.218517   0.258718
         Direct NLinear       0.327724   0.353410
Jena     A                    0.253093   0.275047
         B learned U          0.257653   0.276000
         B learned + Gamma    0.255666   0.271623
         F0                   0.292300   0.264758
         LoRA native          0.240590   0.226584
         LoRA full MSE        0.232892   0.238943
         Direct NLinear       0.273984   0.286714
Hog      A                    2.658552   0.796149
         B learned U          2.707084   0.801910
         B learned + Gamma    2.703102   0.766173
         F0                   2.738044   0.636102
         LoRA native          2.683560   0.618902
         LoRA full MSE        2.625304   0.623919
         Direct NLinear       3.059351   0.775475
```

미압축 MSE-LoRA가 세 자료 모두 최저 combined MSE이고, 기존 native-LoRA가 최저 MAE다. A의 MSE-LoRA 대비 MSE 손해는 Robin 24.97%, Jena 8.67%, Hog 1.27%다. 특히 Hog에서 A가 native-LoRA보다 MSE는 0.93% 낮아도 MAE는 28.64% 높다. 작은 MSE 차이만으로 정확도 문제가 해결됐다고 말할 수 없다. A와 학습 B는 세 자료에서 direct NLinear보다 combined MSE가 낮지만, Hog의 A·Gamma 없는 B는 direct보다 MAE가 높다.

## 각각의 직접 대조에서 남은 변경

A는 `F_phi(X U0) U0^T + q*(X)`이며 기존 학습 LEVEL의 PCA와 Q 예측을 고정하고 내부 q/v LoRA만 학습했다. 미압축 MSE-LoRA는 같은 초기 adapter·학습 목적·LR 후보·선택 기회를 가진 외부 대조다. A와 native-LoRA의 차이를 압축만의 인과 효과라고 해석하지 않는다.

- **Jena A:** 부모 LEVEL MSE 0.324359 → 0.253093, 21.97% 감소. MAE도 0.320450 → 0.275047로 감소했고 두 seed·두 기간 모두 방향이 같다. 조건부 MSE 상대차 95% 블록 구간은 −25.74%~−17.22%다.
- **Robin A:** 부모 0.275196 → 0.273071, 0.77% 감소이나 seed·기간 방향이 다르며 구간 −8.55%~+5.70%다. 안정적 개선 주장은 유지하지 않는다.
- **Hog A:** 부모 2.666809 → 2.658552, 0.31% 감소, 구간 −1.56%~+0.44%다. 외부 대안과의 비용 절충과 A의 신규 내부 적응 효과를 구분한다.

B는 `S*(X) + [F0(X U) − S*(X) U] Gamma U^T`다. S*는 원채널 직접 NLinear, F0는 동결이다. Gamma=I에서 U만 학습한 효과를 먼저 비교한 뒤, 선택된 fixed/learned U 각각에만 VAL Gamma를 적합했다.

- **Jena 공간 학습:** fixed U 0.328154 → learned U 0.257653, MSE 21.48% 감소, MAE 14.59% 감소. 조건부 MSE 구간은 −24.65%~−17.63%이고 두 seed·기간 방향이 같다.
- **단순 혼합의 반론:** fixed U에 Gamma만 적합해도 0.265937까지 감소했다. 학습 U+Gamma 0.255666은 fixed U+Gamma보다 3.86% 낮아 추가 효과가 남지만, B의 큰 개선 전체를 공간 학습만의 기여로 돌리지 않는다.
- Robin/Hog의 learned U 개선은 각각 2.17%/1.60%이며 조건부 구간이 0을 포함한다. Robin learned Gamma는 Gamma=I보다 MSE/MAE를 오히려 높였다. Hog fixed Gamma는 MAE를 낮추면서 MSE를 높였다. Gamma를 항상 유리한 후처리로 채택하지 않는다.

Gamma 전체가 0인 경우는 없었다. Robin seed92602 fixed/learned에 정확한 0 방향이 하나씩 있어 해당 실행만 생략했다. Jena/Hog는 모든 K를 실행한다. 작은 양의 Gamma를 계산 절감으로 세지 않았다. U 학습은 부분공간 변화와 내부 회전·부호 적응을 함께 포함하며, 어느 하나의 원인 증명은 아니다.

블록 구간은 선택된 두 seed·부모에 조건부이며 반복 VAL의 S/U/LR/Gamma 선택 불확실성을 모두 포괄하지 않는다. [선택 봉인](selection_seal.json)은 실행 순서 기록이며 새 독립 확증을 만들지 않는다.

## 같은 회차의 배포 절충

FP32·TF32 off·CPU4 threads, 같은 VAL24와 전체 H48 원채널 CPU 반환을 측정했다. 모델 로드·디스크·채점·원장은 시간 밖이며 온라인 예측과 전송·조합은 안이다. 10회 예열, 10 passes, 3개 순서 블록을 모두 보존했다. 아래 처리량은 seed별 블록 중앙값의 평균이고, peak allocated는 모델 상주 메모리를 포함한다. reserved·resident·배포 bytes와 다른 지표다.

```text
Dataset  Execution                 B4 origins/s   Peak allocated MiB
Robin    A full                      207.10          214.521
         Full MSE LoRA full          245.96          265.966
         Full MSE LoRA chunk_4K       93.20          215.022
         Full MSE LoRA chunk_K        27.63          197.560
Jena     A full                      309.48          218.053
         B learned full              306.39          218.095
         Full MSE LoRA full          224.52          282.312
         Full MSE LoRA chunk_4K       92.01          219.336
         Full MSE LoRA chunk_K        28.17          198.605
Hog      A full                      278.59          225.672
         Full MSE LoRA full          160.72          328.677
         Full MSE LoRA chunk_4K       88.87          225.495
         Full MSE LoRA chunk_K        24.41          199.949
```

Jena의 A는 미압축 MSE-LoRA의 full 실행보다 오차가 높지만 처리량·할당 메모리에서 절충을 만든다. 소배치 LoRA도 비슷하거나 더 낮은 메모리에 들어가지만 이번 조건에서는 더 느리다. Hog에서도 MSE 기준 비용 절충이 있으나 큰 MAE 손해와 부모 대비 작은 증분 효과가 남는다. Robin은 A full이 LoRA full보다 느리므로 일괄 속도 개선 주장을 하지 않는다. 직접 NLinear는 훨씬 작고 빠른 대안이며 낮은 비용보다 예측 오차를 우선할 이유가 있는 경우에만 TSFM 후보를 선택할 수 있다.

할당 메모리 1MiB 안팎 차이를 배포 결정의 확정 경계로 만들지 않는다. 예를 들어 Hog LoRA chunk_4K는 A full보다 할당 메모리가 조금 더 낮다. reserved의 순위와 activation만의 순위로도 바꾸지 않는다. 전체 K 전선이 아닌 고정 K의 측정 선택지다.

Robin B1 첫 블록 F0 sentinel은 앞뒤 지연이 19.79% 변했다. 모든 원반복·블록을 남겼고 Windows·온도·clock을 원인으로 확정하지 않았다. B1 및 작은 비용 차이의 안정적 우위는 제한하며 B4도 seed×block 범위를 함께 읽는다. 범위는 신뢰구간이 아니다. 과거 회차 timing과 나눈 속도비는 사용하지 않았다.

B1 A의 Robin/Jena/Hog 지연은 10.910/10.810/10.817ms, 미압축 MSE-LoRA full은 10.805/10.830/11.303ms다. 최고 단일 원점 지연 주장은 지지되지 않는다. Robin A의 B4 블록 원점당 시간 범위도 2.812~5.348ms로 넓어 평균값만으로 안정적 가속을 주장하지 않는다. B4 처리량은 시간 평균의 역수를 새로 계산하지 않고 저장된 처리량 집계값을 사용했다.

CPU 직접 NLinear의 B4 처리량은 Robin/Jena/Hog 약 41,173/40,363/37,830원점/초, B1은 0.0490/0.0501/0.0521ms다. 이는 별도 CPU 배포 대안이며 GPU 커널 비교가 아니다. GPU 직접 NLinear는 각각 7,109/6,280/4,282원점/초와 9.50/9.56/9.74MiB다. CPU의 CUDA 메모리는 0이 아닌 N/A다.

Jena full에서 peak reserved는 A 238MiB, learned B 236MiB, MSE-LoRA 312MiB이며 allocated와 구분된다. [모든 full/chunk 비용 집계](cost_summary.json), [GPU 원반복](cost_cuda_rows.json), [CPU 원반복](cost_cpu_rows.json)을 함께 보존했다. [정확도 그림](figures/v11_accuracy_by_dataset.png)과 [정확도–비용 그림](figures/v11_accuracy_cost_batch4_v2.png)은 모든 예정 방법을 포함한다. 비용 그림의 처리량 축은 로그이며 막대는 seed×block 최소–최대다. [렌더 수정 기록](figure_revision01.json)은 숫자가 같은 원본 그림도 보존한다.

## 비용 전체와 구현 유효성

A의 신규 등록 LoRA는 294,912개지만 재사용 Q의 G 17,920개 적합 이력이 있다. B는 S* 24,624개, 새 U raw parameter C×K(85/126/256), Gamma K의 단계가 있다. 단계별 등록 수의 합은 실제 변화 scalar 수·함수공간 차원·배포 크기가 아니다. 배포 A는 Q와 하나의 병합 TSFM, B는 S*·export된 고정 U·Gamma와 필요한 TSFM을 포함한다. 작은 학습부를 작은 전체 모델로 부르지 않는다.

신규 fit·역사적 부모/S 적합·선택 탐색은 별도로 기록한다. 다른 시점의 역사적 학습 시간 합을 통제된 학습 속도비로 해석하지 않는다. 학습·복원·추론 실패 시간도 원장에 남긴다. 정규화의 정확한 0분산 backward 수리와 일시적 원장 파일 접근 수리는 [REPAIRS.md](REPAIRS.md)에 원 실패와 검증을 보존했으며 방법의 기여로 세지 않는다.

[적응 비용 표](adaptation_cost.csv)는 선택 fit과 두 LR 전체 탐색, 부모/S 재사용 시간을 분리한다. 예를 들어 Jena 신규 4fit 탐색은 A 789.20초, U 2,931.01초, 미압축 MSE-LoRA 1,198.46초였다. 학습 epoch·목적 경로가 달라 통제된 속도비가 아니며, 작은 U 변수 수가 짧은 학습 시간을 보장하지 않는 실제 반례다. Jena 배포 tensor bytes는 A 190,945,292, B 190,971,628, 미압축 MSE-LoRA 190,872,604, 직접 NLinear 99,000이다. TSFM을 포함한 후보는 전체 모델 저장 크기를 줄이지 않았다.

## 선택과 남은 결정 하나

**A는 Jena에서 확인된 내부 적응 효과와 비용 절충을 중심으로 조건부 유지한다.** Robin의 약한 개선과 Hog의 MAE 손해 때문에 세 자료를 통합한 정확도 해결책으로 채택하지 않는다. B의 U 학습은 Jena에서 fixed 및 fixed+Gamma 대비 추가 효과가 확인된 대안으로 유지하되, 작은 학습 변수 수만으로 A나 외부 LoRA보다 우수하다고 하지 않는다. Gamma는 자료마다 손익이 있어 보정 전·후를 함께 남긴다.

이번 회차는 두 시스템의 직접 비교와 bounded decision을 완료하는 연구다. LoRA·잠재 어댑터·NLinear·예측 혼합의 기존 원리를 인정하며, 조합 자체의 최초성은 주장하지 않는다. 과거 v8/v9/v10보다 좋아졌다는 사실만으로 성공 처리하지 않았고 새 후보·자료·K 탐색으로 넘어가지 않았다.

**다음 결정은 하나다:** A의 Jena 조건부 근거를 중심으로, Q 없이 별도로 학습한 같은 압축 LoRA 대조를 다음 연구에서 추가할지 판단한다. 이는 현재 A의 main-only fit0 결과로 대체할 수 없는 학습정책 대조다. 이번 예비로 자동 실행하지 않으며, 이 비교와 방법 선택을 마친 뒤에야 별도의 미사용 평가를 설계한다.
