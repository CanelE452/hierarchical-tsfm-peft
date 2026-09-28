# V7 practical controls: decision record

**판단: 건물 전력에서 관찰된 수준 보존 잔차 설계의 기여는 조건부 유지하고, 범용 잔차 입력 우위와 고유한 메모리 절감 주장은 축소한다.** 승인된 A와 B 비교는 모두 완료했다. 이 회차가 해결한 것은 직접 선형·소배치 대조의 공백이며, LoRA 대비 남은 정확도 손해나 도메인 일반화 문제를 해결한 것은 아니다. 아래 숫자는 저장 결과에서 확인한 값이다. 최종 검산·게시 상태는 마지막 항목에서 구분한다.

## 확인된 Robin 추가 대조

- 직접 공유 NLinear는 마지막 관측 제거·복원과 bias 포함 Linear(512,48)만 사용한다. 2LR×2seed 4fit에서 TRAIN probe 감소·실제 갱신·VAL 선택 및 복원 정합성을 확인했다. LR=0.001이 두 seed의 best VAL 평균으로 선택됐다. 기존 LEVEL은 다시 학습하지 않았다.
- 전체 masked channel-macro MSE는 LEVEL 0.275196, DIRECT_NLINEAR 0.327724다. LEVEL의 상대 손실 감소는 16.03%이며 두 기간과 두 seed에서 방향이 같다. 이는 v6 뒤 설계한 노출 자료의 추가 대조이고, 모든 선형 방법에 대한 우위나 사전학습만의 인과효과가 아니다.
- F0 0.229095, LoRA 0.223610의 정확도는 더 높다. LEVEL의 LoRA 대비 23.07% MSE 손해는 해결되지 않았다. NLinear는 훨씬 작고 빠르므로, 사용자의 허용 오차가 없는 상태에서 LEVEL을 자동 배포 추천하지 않는다.

## 같은 회차의 실행 대조

GPU 117행과 CPU NLinear 12행을 완료했다. full/chunk/merge 정합성은 고정 atol1e-5/rtol1e-4에서 통과했고, 최대 TEST 절대 차이는 3.248453e-6이다. 전체 과거 512와 미래 48, 원채널과 모델 가중치를 유지했다.

미압축 F0/LoRA의 B4 M20 소배치는 LEVEL 기본 실행보다 낮은 peak allocated를 제공하므로, 압축만이 해당 메모리를 달성한다는 주장은 성립하지 않는다. 다만 이 소배치는 더 많은 순차 호출과 낮은 처리량을 보였다. 원래 미압축 실행은 더 정확하고 메모리가 크다. LEVEL은 이들 사이의 정확도·할당 메모리·시간 절충으로 해석한다. reserved와 allocated는 별개이며, 원시 블록·seed·sentinel 변동과 일부 full 경로 순위 역전을 보존했다. 안정적 일반 속도 우위나 실제 메모리 부족을 주장하지 않는다.

```text
Robin, same-session B4 GPU       MSE       peak allocated MiB   origins/s
LEVEL original                  0.275196  214.489              254.29
LEVEL generic full M20          0.275196  214.493              226.87
F0 original                     0.229095  265.489              205.27
F0 generic M20                  0.229095  212.659               74.73
LoRA original                   0.223610  265.489              204.41
LoRA generic M20                0.223610  212.659               64.39
DIRECT_NLINEAR                  0.327724    9.498            15300.47
```

시간 중심값은 fit별 3블록 중앙값의 중앙값을 구한 뒤 seed 평균이다. 모든 M5/full 경로, batch1 지연, reserved 및 블록 범위는 비용 원본과 본문/부록에 남겼다. NLinear CPU B4는 별도 배포 대안으로 43,879.94원점/초였다. LEVEL 학습부는 17,920개지만 배포 파라미터는 47,736,106개이며, DIRECT는 학습·배포 모두 24,624개다. 작은 학습부를 작은 전체 모델로 부르지 않는다.

## 남은 오류와 선행 경계

정정 진단은 예측 ensemble이 아닌 개별 seed 손실 평균이다. 완전관측 벡터 3838/3840의 PCA 공간 분석은 원래 masked macro MSE와 다르다. LEVEL의 OLD 대비 이득은 버린 공간 Q에서 나타나지만, LoRA 대비 남은 손해는 보존 공간 P와 Q 양쪽에 있다. 일부 채널 집중과 넓은 시점별 손해가 함께 남으며, 과거 잔차 수준·변동과 미래 오차의 관계는 사후 단서다. 이 결과로 이번 LEVEL/K/G를 바꾸지 않았다.

고정 직교 PCA와 공유·무편향 선형 G에서는 LEVEL이 잠재 TSFM에 projected low-rank NLinear형 보정을 더한 것과 같다. 수준 정규화, 잠재 어댑터, 요인과 잔차의 결합 자체를 새 발명으로 세지 않는다. 남는 기여는 특정 정보·공간 제약의 설계와 대응 대조에서 실제 확인한 효과 및 적용 한계다. 이상적 직교성의 대수 성질과 FP32 출력·성능 증거는 구분한다.

## Jena·최종 결정

이전 사용 이력이 있는 Jena 2024 C21/K6, 10분 자료의 12fit과 13개 선택/무학습 모델 평가를 완료했다. L512는 약85시간20분, H48은8시간이며, TEST-A 9–10월·TEST-B 11–12월 각365원점을 사용했다. 독립 확증으로 부르지 않으며 Robin 원MSE와 평균하지 않는다. 두 자료의 모든 fit은 상한 전 정상 정체 종료했고, Jena DIRECT도 LR=0.001을 VAL로 선택했다.

```text
Jena model       combined MSE  combined MAE  TEST-A MSE  TEST-B MSE
LEVEL            0.324359      0.320450      0.422475    0.226243
OLD              0.323305      0.321819      0.422585    0.224024
matched RAW      0.291159      0.306634      0.360648    0.221671
LEVEL_ONLY       0.355720      0.326104      0.462181    0.249258
COMPRESS         0.381187      0.381092      0.486863    0.275510
F0               0.292300      0.264758      0.370756    0.213844
LoRA             0.240590      0.226584      0.285445    0.195735
DIRECT_NLINEAR   0.273984      0.286714      0.332070    0.215899
```

LEVEL의 MSE는 OLD보다0.33%, RAW보다11.40%, DIRECT보다18.39%, LoRA보다34.82% 높다. OLD 대비 차이는 작지만 동등성을 입증한 것은 아니다. 고정 seed·42원점 블록의 combined 상대차95%구간은 OLD대비[0.05%,0.78%], RAW대비[7.52%,14.20%], DIRECT대비[11.86%,24.08%]이며 학습·선택 불확실성까지 포함하지 않는다. ONLY 대비8.82% 개선은 남지만, 이것만으로 잔차 입력의 RAW 대비 추가 가치를 주장할 수 없다. 자료를 바꾸거나 좋은 기간만 선택하지 않았다. Jena 비용을 측정하지 않았으므로 Robin 처리량을 이식하지 않는다.

**유지:** Robin에서 OLD/RAW/ONLY 및 직접 NLinear 대비 확인된 정확도 이득과, 지정된 장치·배치·구현에서의 세 축 절충. **축소:** 모든 도메인의 우위, 압축만 가능한 메모리 절감, 안정적인 일반 처리량 우위, 새로운 수준 정규화라는 주장. Jena의 현재 LEVEL 구성은 단순 대안보다 유리한 선택으로 권하지 않는다.

**다음 결정 하나:** 범용성 확대보다, 건물 전력으로 한정한 고정 적용조건의 외부 재현을 진행할지 판단한다. 이번 회차에서 새 자료·모듈·튜닝을 추가하지 않는다. v7 비교 완료와 논문 신규성·제출 준비·채택은 별개다.

## 근거와 정정

- 정확도: [Robin 평가](robin_eval01.json), [Jena 평가](jena_eval01.json), [전체 수치/분모 연결](report_values.json), [CSV](accuracy_comparison.csv).
- 비용: [전체 요약](cost_summary.json), [GPU 원반복](cost_gpu_rows.json), [CPU 원반복](cost_cpu_rows.json), [정합성](microbatch_parity.json).
- 진단: [정정 결과](robin_diagnostics_corrected_v7.json), [최초 오류와 정정 범위](diagnostic_correction.json). 원본 ensemble 진단은 보존하지만 위 결론의 근거로 쓰지 않는다. 주 정확도 평가는 이 오류의 영향을 받지 않았다.
- 잔차 보정오차: [완전관측 Q 오차와 모든 채널 산점도](robin_q_residual_error_diagnostic_v7.json). 전체 LEVEL 오차와 별도로 계산했으며 인과·routing 근거로 쓰지 않는다.
- 그림: [정확도·비용·P/Q 그림 원본과 hash](figure_manifest.json). 그림의 수치는 결과 파일과 연결되며, 학습 파라미터 수를 과학적 우위로 대체하지 않는다.
- 선행·원고: [기여 경계](CLAIMS_PRIOR_ART.md), [영문 본문](PAPER_DRAFT.md), [부록](APPENDIX.md).
- 원장: [STATUS](STATUS.md), [ledger](ledger.json). 최초 CPU 구문 검사 실행기가 예약 전에 quoting 오류로 끝난 0.247662초를 계상했다. 학습/optimizer 실행은 없었으며 이 예약 순서 위반을 숨기거나 소급 정상화하지 않는다.

실자료16/20attempt, 합성1/6세션(총10update), GPU3354.037초/14400초를 사용했다. 실패한 실자료 fit·연장은 없었고 예비4fit는 미사용이다. [최종 검산](final_checks.json)과 [게시 확인](publication_receipt.json)은 별도의 실제 상태 기록이며, 문서 존재나 실행팀 검산을 독립 재현 또는 과학적 우위로 세지 않는다.
