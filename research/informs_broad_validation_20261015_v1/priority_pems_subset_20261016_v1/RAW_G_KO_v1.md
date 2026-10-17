# PeMS08 C32 Raw·직접 G 검증 결과

상태: `PASS`. 전체 MASTER와 원 subset 전체 실험은 미완료다.

[독립 감사](INDEPENDENT_RAW_G_AUDIT_ROOT_v1.json) · [원정밀도 CSV](RAW_G_RESULTS_v1.csv) · [입출력 SHA](RAW_G_BINDINGS_v1.json)

이번 비교는 ID만으로 정한 C32와 동일 scaler·V148·mask·origin·seed92601의 Raw와 직접 G다. L512/H12이며 주지표는 A 정규화 채널 평균 MSE, 보조지표는 MAE다.

Raw는 압축 없이 C32를 동결 Chronos-Bolt에 입력한다. G는 foundation 없이 채널 공유 512→32→12 GELU innovation과 현재값 anchor를 학습한다. 이 두 경로만으로 B2·압축의 가치나 Bolt 필요성을 확정할 수 없다.

## 실제 정확도·학습·비용

### RAW

```text
정규화 MSE  0.09051378995956519
정규화 MAE  0.19521148986641665
관측 label  56832 / V origin 148
search      0 updates / 0 exposures
선택 refit  epoch=None / 0 updates / 0 exposures
학습 상태   NO_TRAINING
B1 비용    PASS_ACTUAL_PHYSICAL_TRIALS / median=0.20499334996566176 s / p90=0.2095716099254787 s
B8 비용    PASS_ACTUAL_PHYSICAL_TRIALS / median=1.7010823499877006 s / p90=1.7519623898901047 s
```

### DIRECT_G

```text
정규화 MSE  0.08146537281056535
정규화 MAE  0.18577497673500085
관측 label  56832 / V origin 148
search      1577 updates / 25215 exposures
선택 refit  epoch=3.0 / 1911 updates / 30570 exposures
학습 상태   UNDERTRAINED_OR_BUDGET_LIMIT
B1 비용    PASS_ACTUAL_PHYSICAL_TRIALS / median=0.00013439985923469067 s / p90=0.00013621030375361444 s
B8 비용    PASS_ACTUAL_PHYSICAL_TRIALS / median=0.000370049849152565 s / p90=0.0003834899049252272 s
```

같은 반복 개발 V에서 G−Raw MSE 차이는 `-0.009048417148999843`다. 구조와 학습량이 달라 이 차이를 foundation 단독의 인과 효과로 해석하지 않는다.

## 검산 범위와 남은 판단

별도 saved full-V FP64 재채점과 실제 새 CPU6(4+2) 증거를 결속하고, 공개 SSE·SAE·count 14,208행/route를 채널별로 math.fsum 재집계했다. 검사 통과는 독립 일반화나 전체 V 재추론을 의미하지 않는다.

Raw B1/B8의 실제 full 대 row128 출력 parity는 별도 증거다. B8은 full 256streams/1call, row128 256streams/2calls다. parity가 실패하면 그 Raw batch 비용만 미완료로 남기며 G의 독립 경로를 취소하지 않는다.

비용은 CPU2의 실제 physical 입력→채움·정규화→F 또는 G→FP64 inverse·CPU 반환이다. warm10/timed20/separate-stage20을 사용하며, cold model load는 whole trial 밖에서 별도로 기록한다. stage 합이 whole과 같다고 주장하지 않는다.

CPU parameter+buffer resident bytes, 실제 guard RSS, 학습 peak, F 단독 resident는 다른 수치다. native memory API가 기록하지 않은 parameter numel·F 단독 resident는 NA로 남긴다. G에는 F hash를 만들어 넣지 않는다.

단일 seed·반복 개발 V·제한 search와 선택 refit으로 수렴은 미확정이다. C170의 기존 1epoch와 C32의 새 3epoch search는 C만의 공정한 인과 대조가 아니다. upstream 미래 보간·전처리 계보는 UNKNOWN이며 새 prefix 접근만으로 이를 해결했다고 말하지 않는다.

다음 판단에는 원 승인 범위의 같은 C32 B0/B2 등 실제 추가 대조가 필요하다. 이번 Raw/G 결과를 full C170·다른 자료 전체 순위나 B2 주 방법 검증으로 확대하지 않는다.

감사 SHA: `53ac139f74da04a0ef9368ed26b057b5e2dc3addce4ee1333fe09299b8b44405`. CSV/MD는 이 실제 감사에 결속되며 입력 목록 전체는 감사의 `inputs[]`를 원본으로 참조한다.
