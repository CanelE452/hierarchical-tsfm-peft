# v14 decision: local nonlinear gain, no common adoption

게시 후 추가한 [고정 PCA 성분 교체 진단](COMPONENT_DIAGNOSIS.md)은 아래 원래 판정을 바꾸지 않는다. 저장 예측의 P/Q를 미압축 F0·FULL_MSE와 바꿔 본 결과, 도움이 되는 성분은 자료·기간에 따라 달랐고 모든 조합의 합산 MSE 점추정은 FULL_MSE보다 높았다. 이는 원인의 제한된 사후 진단이며 새로운 저비용 모델이나 보호 확증이 아니다. 원본 근거와 승인/선택/평가 기록은 그대로 보존했다.

**판단:** 고정 PCA LEVEL에 raw-history GELU 보정을 붙이는 방법을 네 자료의 공통 해법으로 채택하지 않는다. Jena에서는 부모와 matched linear보다 개선했지만, 직접 예측 대안과 강한 외부 기준선의 정확도에 미치지 못했다. Hog는 악화했고 Robin·Peacock Education은 새 보정이 선택되지 않았다. 이는 이 고정 계약의 개발 결과이며 비선형 보정 일반의 불가능성이나 TSFM 일반의 불필요성을 증명하지 않는다.

근거는 [evaluation01.json](evaluation01.json)의 76개 모델 평가, [selected.json](selected.json), [training_diagnosis.json](training_diagnosis.json)이다. 아래 변화율은 저장된 비교값을 반올림해 옮긴 것으로, candidate/reference에 대한 MSE·MAE 변화율이며 음수가 개선이다. 합산값은 기간별 채널 오차와 관측 수를 합친 뒤 macro 평균하고 seed 손실을 평균한 값이다. 예측 ensemble이나 자료 간 raw 점수 평균이 아니다.

## 결정에 필요한 반례

**Jena — 실제 국소 개선이지만 외부 정확도 격차는 남는다.** LEVEL_GELU는 LR 1e-4, 두 seed의 epoch 29/35를 선택했다. 부모 대비 합산 MSE −12.329%, MAE −6.454%; matched LEVEL_LINEAR 대비 −0.799%, −0.988%다. 선형 대비 조건부 block 95% 상대 구간은 MSE [−1.679, −0.457]%, MAE [−1.543, −0.822]%로, 합산 개선을 단순히 불확실하다고 지워서는 안 된다. 두 seed 모두 부모와 선형보다 합산 MSE·MAE가 낮다. 그러나 test_a 선형 대비 MSE −1.305%와 달리 test_b는 +0.023%이며 구간 [−0.628, +0.332]%가 0을 포함한다. 부모 대비 test_b MSE도 −3.337%의 점추정에 비해 구간 [−5.639, +0.276]%가 0을 포함한다.

Jena LEVEL_GELU는 DIRECT_GELU보다 MSE +5.826%, MAE +6.692%이고, 추가 보정 없는 DIRECT보다도 +3.790%, +4.553%다. DIRECT_GELU 대비 합산 구간은 각각 [3.080, 8.466]%, [3.339, 8.166]%다. 두 seed 모두 직접 대안보다 합산 손실이 높다. FULL_MSE 대비 +22.103%, +25.456%이며 양 기간과 양 seed에서 열세다. 내부 LoRA A 대비도 +12.357%, +8.988%다. F0 대비 MSE 점추정은 −2.714%지만 구간이 0을 포함하고 MAE는 +13.224%이므로 전반적 F0 우위 주장도 성립하지 않는다. 따라서 Jena의 작은 activation 효과와 TSFM 기반 시스템의 실용 우위는 별개다.

**Hog — VAL 개선은 노출된 평가 구간으로 이어지지 않았다.** LEVEL_GELU는 LR 1e-4, epoch 12/14를 선택했고 양 seed의 고정 TRAIN probe와 VAL MSE는 부모보다 낮았다. 그러나 합산 평가에서는 부모 대비 MSE +4.681%, MAE +2.367%로 악화했다. 상대 구간도 [2.486, 7.638]%, [1.101, 4.165]%다. 두 seed 모두 악화했고 test_a/test_b MSE 점추정은 +1.819%/+6.717%로 후반 악화가 더 크다. matched linear 대비 +2.198%, +1.207%; MSE 구간은 0을 포함하지만 MAE 구간 [0.060, 2.993]%는 악화 방향이다. 선형 보정 자체도 부모 대비 +2.430%, +1.146%여서 GELU만 제거하면 일반화 문제가 해결된다는 결론은 아니다.

Hog FULL_MSE 대비 합산 MSE +6.336%의 구간은 0을 포함하지만 MAE +31.691%는 분명한 반례다. test_a MSE 점추정은 −4.088%로 방향이 반대이고 test_b는 +14.800%다. LEVEL_GELU가 DIRECT_GELU보다 MSE는 −7.187%라도 MAE는 +9.391%이므로 한 지표만으로 직접 대안을 배제하지 않는다.

**Robin·Peacock Education — 추가 head의 학습 효과가 없다.** 두 자료 모두 LEVEL_GELU와 LEVEL_LINEAR가 두 LR·두 seed 전체에서 epoch0 부모를 선택했다. 선택된 새 head의 변경량은 0이다. 새 실행과 저장 부모 점수의 약 1e-9 수준 차이는 수치 재실행 차이이며 개선으로 해석하지 않는다. 두 새 LEVEL family는 양 기간·양 seed에서 서로 동일하다. FULL_MSE 대비 합산 MSE/MAE 격차는 Robin +25.938%/+25.462%, Peacock +2.568%/+5.409%로 남는다. Robin은 기존 GROUP_RES에도 +6.478%/+7.139%로 열세다. Peacock에서 기존 A도 동일 부모 fallback이므로 A와의 미세 차이 역시 새 학습 효과가 아니다. FULL_MSE 대비 MSE 구간이 일부 개별 기간에서 0을 포함하는 사실을 숨기지 않으며, 합산과 기간별 자료는 원본 JSON/CSV에 보존한다.

## 직접 보정 대안과 해석 범위

DIRECT_GELU는 네 자료에서 LR 1e-3의 학습된 두 seed를 선택했다. DIRECT 대비 합산 MSE/MAE 변화는 Robin −11.250%/−6.742%, Jena −1.924%/−2.005%, Hog −1.684%/−3.142%, Peacock +1.781%/+0.526%다. Robin·Jena는 양 seed의 두 지표가 개선했다. Hog의 MSE 합산 구간은 [−3.527, +2.299]%로 불확실하며 test_b에서는 두 지표의 구간이 0을 포함한다. Peacock은 두 seed 모두 MSE가 악화했고 MAE는 seed별 방향이 갈렸다; 합산 두 지표 구간은 0을 포함한다. 즉 작은 비선형 보정의 유용성도 부모와 자료에 의존한다. DIRECT_GELU에 별도의 DIRECT_LINEAR 대조군은 없으므로 직접 부모에서 나타난 개선을 활성화의 고유 효과로 분리할 수 없다.

48개 fit은 정상 완료했으며 고정 patience로 종료했고 epoch cap에 닿지 않았다. 저장 gradient·실제 업데이트·동결·재개 상태 검증과 checkpoint replay는 학습이 수행됐다는 근거다. Robin·Peacock의 LEVEL_GELU에서는 고정 TRAIN probe의 최종 손실이 내려가도 VAL 최저값은 초기 부모였으므로, epoch0 선택을 무학습 실행 오류로 취급하지 않는다. 다만 probe는 전체 TRAIN 평가가 아니며 정상 종료가 최적화의 전역 최적성, 더 긴 학습의 무용성, 오류의 인과적 원인을 증명하지도 않는다. Jena/Hog의 선택 차이만으로 P 또는 Q 하나를 남은 격차의 보편적 원인으로 지정하지 않는다. 과거 진단의 공통 GROUP 좌표와 이번 고정 PCA 부모의 구분은 [PLAN_ERRATUM.md](PLAN_ERRATUM.md)에 따른다.

모든 비교는 이미 노출된 네 개발 자료와 선택된 두 seed에 조건부다. 2,000개 paired block 구간은 기간 경계를 넘지 않지만 역사적 모델·방법 선택, LR 탐색 전체, 새로운 자료로의 일반화 불확실성을 포함하지 않는다. 공동 selection seal은 이번 TRAIN/VAL 선택 순서를 고정하며 새 독립 확인으로 바꾸지 않는다. 자체 검산은 독립 재현이 아니다.

## 같은 회차 비용과 최종 선택

[cost_decision.json](cost_decision.json)의 사전 운영 규칙을 충족한 자료는 **Jena만**이다. 이는 유의성 또는 과학적 성공 기준이 아니다. 같은 RTX4070/FP32/TF32 off/CPU4threads/VAL24 원점에서 GPU132개·CPU24개 항목을 완료했다. CPU 표준화 입력부터 온라인 부모 실행과 전체 CPU 출력까지 측정했으며 TRAIN/VAL 예측 cache는 배포 시간에 쓰지 않았다. 아래는 seed별 세 block 중앙값을 다시 seed 평균한 값이다. batch1은 원점당 ms, batch4는 원점/초다. 모든 반복·불리한 block·reserved 메모리는 [cost_summary.json](cost_summary.json), [cost_comparison.csv](cost_comparison.csv), [cost_cuda_rows.json](cost_cuda_rows.json), [cost_cpu_rows.json](cost_cpu_rows.json)에 남긴다.

```text
Jena model / GPU execution   MSE       MAE       B1 ms   B4 origins/s  B4 allocated MiB
LEVEL_GELU / full            0.284368  0.299768  13.184       241.16           218.04
LEVEL_LINEAR / full          0.286657  0.302761  13.177       209.75           218.04
PCA_LEVEL / full             0.324359  0.320450  12.122       204.06           217.97
A / full                     0.253093  0.275047  12.291       240.71           217.97
F0 / full                    0.292300  0.264758  11.820       215.00           281.94
FULL_MSE / full              0.232892  0.238943  12.687       189.99           281.94
FULL_MSE / chunk 4K          0.232892  0.238943     N/A        76.73           218.34
FULL_MSE / chunk K           0.232892  0.238943  47.015        24.88           197.89
NATIVE / full                0.240590  0.226584  12.068       210.64           281.94
DIRECT_GELU / direct         0.268713  0.280965   0.453      2220.55             8.81
DIRECT / direct             0.273984  0.286714   0.256      5139.65             8.56
```

FULL_MSE chunk4K는 사전에 batch4에만 지정했으므로 B1 값이 없다. CPU 대안도 별도 측정했다: DIRECT_GELU batch4는16,408.14 원점/초, batch1은0.15580ms이며 DIRECT는31,066.21 원점/초와0.07135ms다. CPU 값을 GPU kernel 속도로 섞지 않는다. 모든 merge/export/full/chunk/CPU-GPU 정합성 검사를 통과했다. 동등성 허용오차 안의 실행 차이가 원래 평가 수치와 동일한 예측 역할을 보존하며, bit equality를 주장하지 않는다.

**실용 반론은 해소되지 않았다.** Jena의 DIRECT와 DIRECT_GELU는 LEVEL_GELU보다 MSE·MAE가 낮고 이 회차 GPU 메모리·시간도 크게 작다. 기존 A도 정확도는 더 좋고 같은 K의 메모리와 batch4 처리량 점추정은 비슷하다. FULL_MSE는 더 정확하며 작은 chunk로 메모리를 더 줄일 수 있지만 순차 호출 때문에 느려진다. 따라서 GELU 보정의 국소 개선이 새로 채택할 TSFM 선택지를 확보했다고 보고하지 않는다.

F0 sentinel의 같은 block 전후 시간 변화는 batch1 약−7.26%~+10.28%, batch4 약−2.73%~+23.23%다. 원인을 Windows/온도/clock 중 하나로 확정하지 않는다. 새 head를 더한 LEVEL_GELU가 부모보다 빠르게 보이는 점추정도 실행 효율 개선의 증거로 쓰지 않는다. 세 passes/block의 제한과 변동을 보존하며 안정적 속도 배수·완성된 Pareto frontier를 주장하지 않는다. 과거 v13과 새 시간을 나눈 비율도 없다. Robin/Hog/Peacock의 새 v14 비용은 측정하지 않았으므로 이식하지 않는다.

새 학습 변수는 세 family 모두17,920개다. 부모까지 누적하면 LEVEL35,840개, DIRECT42,544개지만 전체 배포 상태는 크게 다르다. Jena LEVEL_GELU는47,754,108개 parameter와191,017,476 tensor bytes(그중 buffer1,044 bytes), DIRECT_GELU는42,544개와171,184 bytes(buffer1,008), DIRECT는24,624개와99,000 bytes(buffer504)다. 미압축 병합 LoRA는47,718,016개와190,872,604 bytes(buffer540)다. 등록 학습부·실갱신·전체 배포 크기·allocated/reserved를 같은 개념으로 쓰지 않는다. 새 head 학습 시간/메모리는 부모 예측 cache를 사용하는 증분 단계이며, 부모 적합의 역사적 비용을 없던 것으로 만들지 않는다.

## 완료 범위와 남은 결정

[final_checks.json](final_checks.json)의 세 검사 묶음이 모두 통과했다:48fit의 실제 초기 tensor·208개 대응 표본 schedule·공동 선택,76개 저장 예측의228개 모델/기간 지표,156개 비용 항목과 보고 표·그림 hash다. 두 그림은 별도로 눈으로 확인했다. 새 모델을 재학습한 독립 재현이 아니며 조건부 bootstrap 구간 전체를 별도 재계산한 검사도 아니다.

GPU 작업 총4,623.910초(약77.1분),48/52 실자료 attempt, 기술 재시도0, 합성1세션/8updates다. cache820.967초·head 학습/VAL3,201.814초·평가 예측175.870초·GPU 비용425.258초를 한 번씩 합산했다. 최종 CPU/저장/게시 수치는 [STATUS.md](STATUS.md)와 원장·게시 receipt를 따른다. 새 raw/백본 다운로드·전역 환경 변경·다른 프로젝트 종료는 없다. 구체적 게시 성공은 원격 SHA 확인 뒤에만 기록한다.

현재 결론은 **이 GELU 추가 단계를 공통 방법으로 채택하지 않고, 같은 질문의 더 큰 head·활성화·LR 탐색을 종료하는 것**이다. Jena의 선형 대비 작은 조건부 이득은 유효한 개발 관찰로 유지한다. 그러나 지금의 단순 대안보다 이 구조를 선택해야 할 근거는 확보하지 못했다. 남은 결정은 이 압축 구조의 추가 개발을 정당화할 구체적 적용 조건이 실제로 있는지다. 그런 조건이나 새로운 근거 없이 모듈을 계속 붙이지 않는다. 기존 결과를 더 넓은 독립 확증으로 바꾸거나 전체 정확도·일반성 문제가 해결됐다고 선언하지 않는다.
