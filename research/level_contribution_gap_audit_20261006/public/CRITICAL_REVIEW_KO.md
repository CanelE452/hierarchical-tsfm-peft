> 공개 사본: 개인 경로·채팅 식별자와 링크를 정리했습니다. 원본 및 사본 해시는 [publication_manifest.json](publication_manifest.json)에 분리 기록합니다. 아래 상태·검산은 원래 감사/실행 시점의 기록이며, 이번 게시 검증이 아닙니다.

# 원래 고정 PCA LEVEL: 기여와 증거 감사

2026-10-06. 대상: CanelE452/hierarchical-tsfm-peft, main `2154e7f2b0e9878164e77536b8deaa9f372ac760`와 로컬 미게시 보충 결과. 첨부 지시문을 사용자가 채택한 범위는 기존 증거 감사이며, 문서에 보존된 과거 예산·후보 학습안을 새 실행 권한으로 해석하지 않았다.

[주장별 증거 색인](evidence_index.csv) · [현재 검사와 읽기만 한 기록](verification.json) · [다음 결정](NEXT_ACTIONS.md).

## 판단과 사용자의 다음 결정

**[판단] 원래 LEVEL의 압축 경로 보완 효과와 일부 건물 전력 자료에서의 잔차 입력 제한 효과는 유지할 수 있다. 그러나 현재 증거만으로 독립적인 새 PEFT 방법론 기여가 확보됐다고 선언하기에는 부족하다. 범용적 실용 우위 주장은 축소하고, 원형을 계속 확장하는 작업은 보류하는 것이 타당하다.**

이 판단은 모든 자료에서 최고 성능을 요구해서 나온 것이 아니다. 제한된 기여에도 의미가 있다. 문제는 현재 남는 효과가 알려진 잠재 어댑터와 선형 외부 보정의 조합을 넘는 **예측 가능한 적용 조건·배분 근거·표준 대안 대비 선택 이유**로 아직 연결되지 않았다는 점이다. 실패 반례가 있는 자료를 빼거나 후속 A/B/GROUP의 결과로 원형을 바꾸면 이 공백이 해소되지 않는다.

이번 감사는 신규 fit, optimizer update, PCA 적합, model forward, GPU 작업, 데이터·가중치 다운로드 모두 0회다. 새 성능 재채점·bootstrap·P/Q 교체 진단도 하지 않았다. 완료 판단은 기존 실제 증거와 남은 공백을 연결한 상태를 뜻한다. 학술적 최초성·채택 여부의 확정이 아니다.

## 무엇을 확인했고, 무엇을 읽기만 했는가

- **이번 직접 확인:** 로컬/원격 Git 상태, 관련 프로세스·앱 채팅 상태, 실제 파일 존재·명시된 크기·제한적인 hash, 기존 manifest/selection/run 기록의 연결, 신규 Chronos 예측의 형상·원점·유한성 등. 상세 범위는 `verification.json`과 세 artifact receipt JSON에 있다.
- **기존 실행의 증거:** 각 회차 PLAN/프로토콜, source, run result/curve, selection seal, checkpoint/prediction manifest, 평가 및 비용 원행. 이것들이 연결된 경우 README의 완료 문장만 읽은 경우와 구분했다.
- **읽기만 한 검산:** 과거 `final_checks` 등의 PASS와 저장 점수 재계산·forward parity 결과. 그 검사를 이번 감사에서 전면 재실행하지 않았다. 실행팀 자체 검산이며 독립 재현이 아니다.
- **현재도 미확인:** 새 자료·다른 백본에 대한 외적 확증, 모든 선행 방법의 완전한 재현, 사전학습 자료 비중복, 단일 실패 원인의 인과 분리.

같은 선택 seed/checkpoint의 후속 재사용은 동일 증거다. F0·zero-shot 출력을 두 학습 seed로 복제하지 않는다. step0 부모 선택은 새 적응 이득이 아니다. v15의 검산 PASS도 누락 비용이 있는 전체 회차 완료를 뜻하지 않는다.

## 비교 계약과 모델 식

[확인] 현재 발표 세 자료는 Robin C17/K5, Peacock Education C13/K4, Jena C21/K6이며 입력512·예측48이다. 원래 TEST-A/B 원점은 각각 40/40, 40/40, 365/365다. Robin/Peacock은 시간 간격, Jena는 10분 간격이다. TRAIN 통계로 표준화하고 기존 결측 입력 대체를 유지한다. 원래 관측 target mask로 채널별 오차합/관측수를 기간 간 합산한 뒤 채널-macro 손실을 구하고, 학습형 모델은 두 선택 seed의 **손실**을 평균한다. 예측 앙상블이나 서로 다른 자료의 MSE 평균으로 총승리를 만들지 않는다.

[확인] v7에서 고정 직교 U, 채널 공유·무편향 선형 G에 대해 다음 조건부 식을 이미 기록했다. Q는 attention query가 아니라 제외 공간 투영이다.

```text
P = U Uᵀ,  Q = I − P
LEVEL(X) = b(X) + N_G(XQ) = b(X) + N_G(X)Q
N_G: 마지막 값을 제거한 공유 시간 선형 예측에 마지막 값을 복원
```

시간 선형 연산과 채널 투영의 교환으로 얻는 함수 등가식이다. optimizer·초기화·mask·학습 절차까지 표준 NLinear와 같다는 뜻은 아니다. bias, 채널별 G, 비선형 G, 학습형 E/D로 조건 없이 확대할 수 없다. 이 등가 모델을 다시 학습하는 것은 새로운 구조적 대조가 되지 않는다.

현재 기본 LEVEL의 출처는 Robin v6, Jena v7, Peacock v12다. MSE-LoRA는 Robin/Jena v11 `full_lora_mse`, Peacock v12 `full_mse`다. 과거 native quantile-loss LoRA는 별도 기준선이다. 내부 LoRA A, 학습 U/Gamma B, GROUP, v14 보정, v15 decoder, v16 temporal은 기본 LEVEL로 합치지 않는다.

## 이미 답한 질문

**1. 압축만 한 주경로의 부족을 보완하는가? — 답함, 세 고정 과제에 한정.**

```text
자료       압축만 b MSE    기본 LEVEL MSE    직접 NLinear MSE    원채널 MSE-LoRA MSE
Robin      0.977023        0.275196          0.327724            0.218517
Peacock    0.298832        0.203266          0.219884            0.198176
Jena       0.381187        0.324359          0.273984            0.232892
```

[확인·기록값] 주경로 대비 보완 효과는 세 자료에 있다. LEVEL 전체가 직접 NLinear보다 낫다는 관측은 Robin/Peacock에 한정되며 Jena는 반대다. 전체의 차이는 TSFM 사전학습 지식만의 순수 인과효과가 아니다. Peacock b는 기존 A step0의 저장 main-only 출처와 LoRA B=0 확인에 연결된다. 이를 새로운 A 성공이나 독립 b 적합으로 세지 않는다. 원정밀도·기간·seed·분모는 기존 Chronos 비교 `accuracy_comparison.csv`와 reuse manifest에 있다.

**2. G에 RAW 대신 제외 공간 입력을 주어야 하는가? — 자료별로 답함.**

[확인·기록값] Robin v6는 LEVEL 0.275196 대 matched RAW 0.346630이다. Jena v7는 0.324359 대 0.291159로 반대다. Peacock의 GROUP_RAW는 같은 PCA 대조가 아니다. 그러나 로컬 v17의 **matched PCA RAW**가 이미 완료돼 있다: 기본 LEVEL 0.2032661712362121 대 RAW 0.27209153229343336, `(LEVEL−RAW)/RAW` −25.29%, 조건부 95% 구간 [−27.40%, −22.29]%. 두 seed와 A/B 모두 LEVEL 방향이다.

따라서 “Peacock에서 동일 PCA RAW를 아직 안 했다”는 공백은 현재 로컬 결과로 정정된다. 다만 v17은 TEST 노출 후 보충 대조다. v12 최초 확인에 소급해서 넣거나 독립 확증이라고 쓰지 않는다. 유지할 문장은 **“이 잔차 입력 제한은 Robin·Peacock에서 이득, Jena에서 손해였다”**이다. 건물 전력 전체에 대한 규칙은 아니다.

**3. 학습형 E/D를 전혀 안 해봤는가? — 그 비판은 기각. 현재 조건의 직접 비교는 미확인.**

[확인] v1 `ForecastAdapter`의 COMPRESS는 PCA 초기화된 bias 없는 E/D를 예측 손실로 학습하고 G는 없다. 백본만 동결한다. 실제 E/D 갱신, 선택 epoch/step, Electricity K8/K16 체크포인트가 있다. v6 고정 PCA COMPRESS와 다르다. 이 초기 결과를 현재 세 자료의 학습 E/D-only 직접 대조나 AdaPTS 전체 재현으로 부르지 않는다. 로컬 검색 범위와 직접 확인한 예시 체크포인트는 early receipt에 있다.

**4. PCA/U·공간 배분을 바꾸는 대조가 없었는가? — 답함, 최적 배분 원인은 미분리.**

[확인·기록값] v11 B는 직접 NLinear 부모와 직교 U 변경, Gamma 혼합을 이미 대조했다. Jena fixed U 0.328154 → learned U 0.257653, fixed U+Gamma 0.265937, learned U+Gamma 0.255666이다. v13은 같은 K의 TRAIN-only 상관 그룹 평균 기저 RES/RAW 대조이며 Robin 개선·Jena 제한·Peacock 반례가 있다. U를 안 바꿨다는 비판은 틀리다. 부분공간·내부 회전/부호·혼합·잠재 모델 분포가 함께 달라져, 예측가능한 방향 하나를 발견했다는 원인 증명은 아니다.

**5. 보존 공간·비선형성·학습 부족만 해결하면 되나? — 관련 시도는 완료, 충분조건은 지지 안 됨.**

[확인] v8~v10의 보존 공간 보정·단계적 P 보정·전체 TRAIN ridge와 v14의 비선형/동일 크기 선형/직접 NLinear+비선형 대조가 있다. v14 Jena에는 비선형 추가 이득이 있지만 더 가벼운 직접 대안보다 나쁘고, Hog는 악화한다. Robin/Peacock은 부모 step0를 선택했다. 정상 학습·복원이 확인된 음성을 미실행이나 버그로 다시 이름 붙이지 않는다. 정상 종료가 전역 최적화 보장은 아니다.

Jena v8 joint P/Q 0.279714, v9 staged P 0.279884, v10 full-TRAIN ridge P 0.279250로 원형보다 좋아져도 직접 NLinear 0.273984를 넘지 못한다. Robin v9는 실제 갱신 후 epoch0 부모 선택이며 v10 후보 악화 뒤 부모 복귀다. v10 비용은 68/78행 `incomplete_budget`로 최종 우열 미확정이다. CPU ridge도 적합이며 optimizer0이라는 이유로 학습0으로 재분류하지 않는다. 최종 Jena 평가 출처는 `jena_eval02.json`; 이전 실패·partial을 보존했다.

**6. 남은 오차가 어느 공간과 관련되나? — 성분 교체는 답함, 근본 원인은 부분만 답함.**

[확인] v14의 저장 예측 swapP=DP+RQ, swapQ=RP+DQ를 재사용한다. FULL_MSE donor 기준 원형의 MSE는 다음과 같다.

```text
자료       LEVEL      swapP      swapQ      FULL_MSE
Robin      0.275196   0.251730   0.241980   0.218517
Jena       0.324359   0.236170   0.321081   0.232892
Hog        2.666809   2.655130   2.638370   2.625304
Peacock    0.203266   0.202519   0.198928   0.198176
```

과거 입력으로 만든 donor의 저장 예측 조합이며 미래 정답 oracle은 아니다. donor 실행비용이 드는 진단이지 경량 새 방법은 아니다. 특히 Jena의 P 교체 효과가 크지만 이것만으로 입력 제한·좌표 분포/정규화·비선형 처리·최적화 중 하나를 원인으로 확정할 수 없다. 완전관측 P/Q 에너지와 주 masked macro MSE는 다르며 Hog에서는 순위 차이도 있다. 이번에 같은 교체를 반복하지 않았다.

**7. 원채널 소배치·좌표 선택·시간축 압축을 안 해봤는가? — 답함, v15 비용 일부만 미완료.**

[확인] v7 이후 원채널 F0/LoRA full·독립 행 chunk와 직접 NLinear의 같은 회차 비용 비교가 있다. 압축만이 메모리를 줄이는 방법이라는 주장은 기각한다. 작은 chunk의 정확도 보존·낮은 allocated와 느린 처리량을 함께 적는다. v15는 원채널 선택/PCA와 독립 decoder의 정확도 비교가 완료됐지만 GPU 606/612 비용행에 그쳐 Robin 6행이 빠졌다. 전 회차 비용 완료라고 쓰지 않는다. v16은 모든 채널·시간4개 평균의 다른 함수이며 예정 비용·예측 검산이 완료됐다. 어느 자료에서도 full-resolution MSE-LoRA의 pooled MSE·MAE를 따라잡지 못했다. 이 결과를 LEVEL 원형의 성과로 교체하지 않는다.

## Chronos-2: 이번 비교는 실제 완료돼 있다

[확인] 로컬 `research/level_chronos2_controls_v1/`에 두 공식 모델 × MV/UNI × 세 자료의 완료 예측, 원점 계약, preflight, seal, 원장, 평가, 같은 회차 비용 및 검산이 있다. 과거 gap_atlas/internal_adaptation의 사용 이력으로 완료를 대신한 것이 아니다. 현재 관련 실행은 확인되지 않아 이어할 미완료분도 없다. **새 평가·다운로드·GPU 측정을 시작하지 않았다.**

큰 모델 revision `29ec3766d36d6f73f0696f85560a422f50e8498c`, Small revision `ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a`다. 설치 API 소스와 wrapper에서 MV는 같은 원점 C채널을 한 과제로 묶고 다른 원점은 분리한다. `cross_learning=False`는 서로 다른 task 간 공유를 끄며 원점 내부 MV까지 끄지 않는다. UNI는 원점/채널별 singleton task다. 기존 preflight는 실제 group mask와 단독/묶음/순서변경/다른 원점 교란/미래 차단을 기록한다. 그 forward 검사 결과를 이번에 다시 생성하지 않았다.

입력512·예측48, 같은 TRAIN 통계·mask·원점, 실제 native 0.5 분위수, FP32/TF32 off/CPU4를 유지했다. 새 모델 추가 적합·미래 공변량·장문맥·원점 간 sharing이 없다. 선택된 기존 참조를 재사용하며 Peacock A step0와 기본 LEVEL을 구분한다.

```text
자료       LEVEL MSE   C2 MV MSE   Small MV MSE   LEVEL→Small B4 원점/초   LEVEL→Small allocated MiB
Robin      0.275196    0.230150    0.224650       149.5 → 184.0             214.5 → 181.4
Peacock    0.203266    0.199572    0.204274       147.4 → 178.5             210.0 → 164.5
Jena       0.324359    0.253987    0.263155       144.1 → 169.1             217.9 → 195.6
```

[확인·기록값] 위 비용은 같은 RTX4070 회차의 VAL24·10 passes·10 warmups·3블록 B4 요약이다. 과거 회차 시간을 섞은 속도비가 아니다. Robin/Jena에서 Small MV는 낮은 MSE·allocated와 높은 B4 처리량이다. `(LEVEL−Small)/Small` MSE 손해는 22.50% [9.74,43.55]%와 23.26% [15.63,30.56]%다. Peacock은 −0.49% [−3.31,5.54]%로 작은 점추정 이점이고, MAE는 LEVEL 0.315004 > Small 0.301992다. 구간의 0 포함은 동등성·비열등성이 아니다.

Jena B4 reserved는 Small252 > LEVEL238 MiB로 반대다. 모든 메모리 축이나 모든 지표에 대한 지배라고 부르지 않는다. 큰 C2 대비 원형의 비용 절충은 남아도, Small을 제외해 원형의 일반 선택 이유로 만들 수 없다. MSE-LoRA와 C2도 Jena에서 MSE/MAE 방향이 다르므로 절대적인 단일 승자는 정하지 않는다. 사전학습 비용·자료 비중복도 이번 zero-shot으로 해결되지 않는다.

## 가장 강한 반론과 그 반론을 뒤집을 증거

**반론:** 원형은 알려진 잠재 변환·복원 경로에 투영된 NLinear형 외부 보정을 붙인 구조다. b 개선만으로 이 분담이 특별히 적합하다고 할 수 없다. Jena에서 RAW와 직접 NLinear가 더 좋고, 현재 세 자료에서 Small까지 포함하면 원형을 선택할 일반적인 정확도–allocated–처리량 근거가 약하다. U/Gamma/그룹/보정의 자료별 결과도 “큰 분산 방향은 TSFM, 나머지는 G”라는 일반 배분 원리를 입증하지 않는다. 뒤이어 많은 대조를 실행했다는 사실은 이 반론의 해답이 아니다.

이 반론을 뒤집으려면 기존 소수 자료에서 좋은 방향·모듈을 사후 선택하는 것보다 다음 중 주장에 필요한 구체적 증거가 필요하다. **이번 실행 계획이나 승인은 아니다.**

- 적용 조건을 TEST 전에 정하고, 그 조건의 미사용 자료에서 고정 원형의 추가 가치를 확인한다. 현재 반복 노출된 자료의 bootstrap만으로 대체하지 않는다.
- 같은 정보·평가·선택/부모 이력을 공개한 학습 E/D-only와 투영/RAW/직접 대안을 두고, 잔차 분담의 정확도 및 역전파·배포 절충이 남음을 보인다. 유리한 결과만으로 AdaPTS 전체보다 낫다고 확대하지 않는다.
- 실제 사용 제약에서 Small·원채널 chunk·직접 NLinear가 충분하지 않은 이유와 허용 손해를 제시한다. 새 임의 합격선·채널 복제·유리한 자료 검색으로 만들지 않는다.
- 배분의 원인을 주장한다면 부분공간·내부 좌표 회전/부호·정규화/정보 경로가 함께 바뀌지 않는 근거가 필요하다. 현재 성분 교체는 위치 진단까지만 제공한다.

## 아직 답하지 못한 질문과 신규 진단 생략 이유

1. **현재 세 자료에서 학습 E/D-only와 원형의 직접 절충:** 검색 범위에서 미확인. v1 Electricity와 v11 B로 채우지 않는다. 따라서 “예측 학습 E/D보다 잔차 적합에 용량을 쓰는 것이 우월하다”는 주장은 보류한다.
2. **큰 TRAIN 분산과 TSFM의 방향별 추가 이득의 관계:** 원인 증명 없음. v11/v13/v14로 고정 PCA의 최적성·보편 배분 주장에 제한이 충분하다. 새 방향별 저장 진단이 관측 상관을 추가해도 Small 대조나 인과/외적 공백을 해결하지 못하므로 이번에는 **해당 없음: 판단을 바꾸지 않아 생략**했다.
3. **미사용 자료에서 예측 가능한 적용 범위:** 현재 후속 결과로 답할 수 없음. Peacock 최초 확인의 범위는 보존하지만 A 내부 적응 이득을 증명한 결과는 아니다. 이후 v13/v14/v17/C2는 이미 노출된 비교다.
4. **v15 Robin의 전체 비용:** 일부 실행만 완료. 다른 회차 비용으로 메우지 않는다. 지금 원형 판단에 이미 충분한 같은 회차 비용이 있어 추가 측정하지 않았다.
5. **백본 일반성·pretraining 순수 효과·최적화의 전역 충분성·모든 선행 대비 최초성:** 미확인. 정상 실행/PASS와 독립 연구 증거를 구분한다.

학습 E/D 공백은 존재하지만, 새 대조 한 번이 고정 PCA의 원인·외적 확증이나 현재 Small 반례를 해소하지 않는다. 현재 선택할 중요한 주장을 E/D 우위로 바꾸지 않았으므로 승인용 신규 fit 계획은 이번 산출에서 제안하지 않는다. 나중에 그 좁은 질문을 별도 연구 목적으로 선택한다면 그때 실제 기준 계약·잔여 예산을 고정한 최소 계획과 승인이 필요하다.

## 원기록의 실패·정정·회계 한계

- v2의 실제 neural20+synthetic2=22/20 attempt 위반과 v7 `PASS_WITH_RECORDED_ACCOUNTING_DEVIATION`을 정상 계약 준수로 덮지 않았다.
- v8의 `PASS_WITH_DOCUMENTED_CONTRACT_CORRECTION` 및 v10의 실패/partial·최종 eval02 출처를 구분했다.
- v13의 사전예약 없는 5.1755초 검사 실행은 공개·사후 계상된 이탈이다. 총예산 초과로 단정하지 않는다.
- v14 성분 진단의 원 PLAN 정정과 canonical PCA 좌표, v13의 GROUP 좌표를 구분했다. v15의 검산 PASS에는 전체 비용 미완료가 명시돼 있다.
- Chronos preflight는 두 체크포인트×세 자료×MV/UNI의 첫 4 VAL 원점을 검사했다. 모든 TEST attention mask를 개별 검사했다고 확대하지 않는다. 전체 VAL 복원 보완은 TEST 이후였고 재선택은 없었다. Peacock b/F0의 저장 VAL 배열 부재는 남는다.
- 이번 Chronos hash 검사의 집계 stdout은 남지만 개별 actual digest 출력은 보존되지 않았다. 예상 digest를 실제 재계산 값으로 복원하지 않고 receipt의 `actual_sha256=null`과 집계 검사 범위를 분리했다. 후반 검사 참조 수는 원필드 합계 642로 정정했으며 참조 수 자체를 모델·실험 수로 세지 않는다.

이 한계들은 원기록과 함께 보존한 감사 결과다. 신규성의 긍정·부정 결론을 임의로 만드는 점수로 쓰지 않는다.

## 선행과 읽은 깊이

2026-10-06에 1차 출처를 재조회했다. 아래 대조는 겹치는 원리와 정보/목적 차이를 설명하는 것이며 완전 문헌조사나 해당 모델의 새 재현 실험은 아니다.

- **AdaPTS:** 공식 ICML2025 출판/초록과 저자 `LinearAutoEncoder` transform/inverse-transform, supervised `make_predictions` 및 예측/선택 손실 경로를 읽었다. 잠재 투영→독립 FM→복원은 겹친다. 선택적으로 더하는 입력 재구성 손실은 별도 미래 제외 공간 예측과 다르다. 읽은 경로 밖의 모든 버전까지 없다고 주장하지 않는다. v1 bias 없는 E/D나 고정 b는 확률적/다양한 어댑터를 포함한 AdaPTS 전체가 아니다. [공식 출판](https://proceedings.mlr.press/v267/benechehab25a.html), [어댑터 코드](https://raw.githubusercontent.com/abenechehab/AdaPTS/main/src/adapts/adapters.py), [예측·적합 코드](https://raw.githubusercontent.com/abenechehab/AdaPTS/main/src/adapts/adapts.py).
- **NLinear/side-tuning:** 저자 NLinear 모델 전체의 마지막 값 제거·공유 affine·복원과 side-tuning 공식 연구 페이지 초록을 읽었다. 외부 경량 경로와 합산은 알려진 원리다. side-tuning의 다른 도메인 실험을 LEVEL의 증거로 쓰지 않는다. [NLinear 코드](https://raw.githubusercontent.com/cure-lab/LTSF-Linear/main/models/NLinear.py), [Side-Tuning](https://sidetuning.berkeley.edu/).
- **ForeCA:** 공식 ICML2013 초록/출판 정보를 확인했다. 시간 의존 신호의 예측가능성을 위한 차원 축소라는 목적은 선행이다. 알고리즘 전체·실험을 이번에 읽거나 실행하지 않았으며 LEVEL과 동치라고 하지 않는다. 분산과 예측가능성의 차이 자체는 새 발견이 아니다. [공식 출판](https://proceedings.mlr.press/v28/goerg13.html).
- **최근 경계:** Time-PEFT(ICML2026)의 시간/다채널 복잡도 및 frequency/channel adapter, CoRA(ICLR2026)의 저랭크 상관 적응, ORCA의 black-box online 잔차 적응은 공식 출판/저자 초록 수준에서 확인했다. 각각 backbone/학습/온라인 정보 계약이 달라 원형과 동일하다고 하지 않는다. 이들의 실험·전체 코드를 재현하지 않았다. 최근 검색에 같은 이름이 없다는 사실도 최초성의 증거가 아니다. [Time-PEFT](https://proceedings.mlr.press/v306/na26b.html), [CoRA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ae3e173398d6e43fea63cbcc16fbaa98-Abstract-Conference.html), [ORCA 저자 원문 초록](https://arxiv.org/abs/2606.14222v2).
- **최신 네이티브 대안의 범위:** Google의 2026-08-31 TimesFM-3 공식 발표도 검색에서 확인했다. 따라서 Chronos 비교 완료를 현재 모든 네이티브 대안에 대한 전수 우위로 표현하지 않는다. 이 추가 모델의 LEVEL 계약 평가는 이번 목표에 넣거나 실행하지 않았다. [공식 발표](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/).

## 유지·축소·보류할 주장

**유지/조건부 유지:** 이 세 계약에서 b의 오차를 제외 공간 예측으로 보완했다. Robin·Peacock matched RAW보다 잔차 입력 제한이 유리했으며 Jena는 반례다. 공유 무편향 G의 투영 NLinear형 식, 작은 등록 적합 변수 수, 같은 회차에서 관측된 특정 비용 절충은 유지한다. 등록 변수 수 감소를 배포 모델 크기·시간·전체 메모리 감소율로 바꾸지 않는다.

**축소:** 건물 전력 전반의 일반성, 원채널 full만 비교한 실용 우위, 큰 C2만 비교한 비용 선택 이유. 모든 chunk·Small·direct 대안과 실제 정확도 손해를 같이 적는다. 기존 검산 PASS나 발표 완료도 연구 신규성의 증거로 쓰지 않는다.

**보류:** 독립 새 PEFT 방법론 주제 확보, 고정 PCA의 최적 배분, 큰 분산이 TSFM 이득의 원인, 학습 E/D-only보다 우월한 용량 배분, 백본/도메인 일반성, 원형의 전반적 우위. “아직 이 모듈을 안 해봤다”는 이유로 v18 이후를 자동 추진하지 않는다.

**기각/정정:** 학습 E/D·U·비선형·원채널 소배치·시간 압축을 전혀 안 했다는 문장, Peacock 같은-PCA RAW가 로컬에도 없다는 문장, A step0를 내부 적응 성공으로 세는 문장, v15 전체 비용 완료, 압축만 메모리를 줄일 수 있다는 문장, 신규 fit 수·검산 PASS로 신규성을 주장하는 문장.

이미 알려진 것은 잠재 어댑터·마지막 값 제거/복원·외부 합산·예측가능성을 고려한 차원 축소다. 현재 기록에서 확인된 것은 제한된 자료의 압축 보완 효과, 자료에 따라 달라지는 잔차 입력 이득과 여러 실패 경계다. 가장 강한 실용 반론은 같은 계약의 Small MV와 chunk/direct 대안이다. 이들로 충분하지 않은 일반적인 사용 조건은 미확인이다. 적용 조건·배분 근거·직접 E/D 및 외적 확인의 공백을 유지한 채 독립 방법론 주제 확보를 선언하지 않는다.
