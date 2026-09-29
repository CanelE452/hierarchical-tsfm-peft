# TOPIC_DECISION — v9 Staged P-Space Correction

상태: **예정된 비교 완료. Jena의 조건부 P 보정 효과는 유지하되, 주력 구성의 전면 교체는 보류한다.** Robin에서는 부모 복귀가 선택됐고, Jena에서는 부모·matched RAW보다 개선했지만 v8보다 나아지지 않았으며 직접 선형·LoRA의 반례가 남았다. 같은 회차 비용도 완료했다. v9는 후속 방법 분석과 절제 비교로 남기며 정확도 손해·범용성 해결로 선언하지 않는다.

근거 파일은 `robin_eval01.json`, `jena_eval01.json`, `analysis01/role_scores.csv`, `analysis01/model_scores.csv`, `analysis01/comparisons.csv`, `analysis01/space_scores.csv`, `analysis01/q_preservation.csv`, `analysis01/q_prediction_parity.csv`, `cost_decision.json`이다. Robin과 Jena는 이미 노출된 개발 평가이므로 독립 확증이 아니다. 모든 전체 수치는 seed ensemble이 아니라 seed별 손실의 평균이다.

아래 수치는 저장 결과에서 확인한 [확인] 값이며 반올림했다. 효과의 원인·다른 집단으로의 전이는 추정 또는 미확정이다. [report_values.json](report_values.json), [학습 요약 표](training_summary.csv), [비용 원반복](cost_jena_gpu_rows.json), [비용 집계](cost_jena_summary.json)로 연결된다. 전체 오차는 각 seed·채널의 기간별 오차합/관측수를 합산한 뒤 채널평균과 seed평균을 취했다. 블록 구간은 두 seed를 고정한95% 조건부 구간이며 Robin7/Jena42원점·기간 내부2,000회 재표집이다. 부모·방법·alpha 선택 전체의 불확실성을 포함하지 않는다.

## 방법과 비교 질문

v9는 새 TSFM 경로를 만들지 않고, 선택된 LEVEL 부모를 그대로 고정한 뒤 작은 시간 head 하나만 추가한다. 입력 \(X\), 고정 부모 \(f^*(X)\), TRAIN PCA basis \(U\), \(P=UU^\top\), \(Q=I-P\)에 대해 두 후보는 다음이다.

```text
STAGED_P   = f*(X) + alpha H_P(D(E(X)) - last(D(E(X))))
STAGED_RAW = f*(X) + alpha H_R(X      - last(X))
alpha ∈ {0, 0.25, 0.5, 0.75, 1}
```

두 head는 모두 bias와 activation이 없는 공유 시간 선형 512→32→48이다. 이번에 새로 학습되는 파라미터는 head 하나의 17,920개뿐이다. 과거에 학습된 부모 LEVEL의 G까지 역사적 adapter 비용으로 합산하면 35,840개지만, v9의 증분 학습 파라미터와 혼동하지 않는다. 배포에는 동결 TSFM 부모가 그대로 필요하므로 “작은 head만 배포”라고 쓰면 안 된다.

8개 적합 모두 실제 head 갱신·부모 parameter/buffer 불변·optimizer 제외·checkpoint/재시작 복원을 확인했다. 모두 VAL 정체6회로 종료했고120epoch 상한에 걸린 실행은 없었다. 정상 업데이트 뒤 step0을 고른 경우와 학습이 끊긴 경우를 구분한다. 고정 TRAIN probe와 전체 VAL, 첫 gradient·최종 weight 검사는 실행별 `runs/<id>/result.json` 및 `curve.json`에 있다.

```text
자료/방법          seed     완료epoch  basic선택epoch  alpha
Robin STAGED_P    92601     6          0               0
Robin STAGED_P    92602     6          0               0
Robin STAGED_RAW  92601     6          0               0
Robin STAGED_RAW  92602     8          2               0.5
Jena  STAGED_P    92601    14          8               1
Jena  STAGED_P    92602    22         16               1
Jena  STAGED_RAW  92601    10          4               1
Jena  STAGED_RAW  92602    22         16               1
```

basic은 step0 포함 alpha1 VAL 최소이며, 보조 alpha는 epoch≥1 중 최선 checkpoint 한 개에만5개 값을 적용했다. 두 자료의8개 학습과 두 선택이 모두 끝난 뒤 TEST를 개봉했다. 부모 선택에 이어 VAL을 재사용한 편향은 남고, alpha0 선택이 미래 비열화를 보장하지 않는다.

핵심 질문은 세 가지다. 첫째, 부모 LEVEL 위에 P-space 보정을 추가하면 부모보다 나아지는가. 둘째, 같은 추가 head 기회를 받은 STAGED_RAW보다 나은가. 셋째, v8 JOINT_PQ, RAW64, DIRECT_NLINEAR, F0, LoRA 같은 가까운 대안 앞에서도 선택 가치가 남는가. 이 결과는 v8의 손해를 “순수 P/Q gradient collision”으로 증명하지 않는다. v8과 v9는 부모 고정 여부, Q 업데이트 여부, checkpoint 선택, scheduler와 clip까지 함께 다르다.

Side-Tuning의 frozen base+side path와 단계적 잔차 적합·축소는 가까운 선행이다. v9의 주장 범위는 고정 LEVEL 부모, P 제한 head, matched RAW, 유한 VAL alpha를 결합한 실용 절차와 그 직접 비교다. 새 가산 원리·정규화·최초성을 주장하지 않는다. 실제 읽은 판본과 깊이는 [PRIOR_ART.md](PRIOR_ART.md)에 구분했다.

## 정확도 결과

전체 결과는 자료별로 갈렸다. Robin에서는 STAGED_P가 부모 LEVEL과 정확히 같은 전체 손실로 돌아갔다. basic STAGED_P도 부모와 같은 예측으로 평가됐고, alpha 정책은 두 seed 모두 alpha=0을 선택했다. STAGED_RAW는 basic에서 한 seed가 부모로 남고 다른 seed가 악화됐으며, alpha 정책도 seed 92601은 alpha=0, seed 92602는 alpha=0.5로 후퇴했다. 반대로 Jena에서는 두 STAGED_P seed 모두 alpha=1을 유지했고 부모와 matched RAW보다 낮은 MSE를 냈다.

```text
dataset  role              MSE       MAE       note
Robin    LEVEL parent      0.275196  0.324593  frozen parent
Robin    STAGED_P          0.275196  0.324593  parent exact
Robin    STAGED_P alpha    0.275196  0.324593  alpha 0,0
Robin    STAGED_RAW        0.276520  0.326018  worse than parent
Robin    STAGED_RAW alpha  0.275487  0.324883  alpha 0,0.5
Robin    F0                0.229095  0.262074  stronger MSE/MAE
Robin    LoRA              0.223610  0.256347  strongest here

Jena     LEVEL parent      0.324359  0.320450  frozen parent
Jena     STAGED_P          0.279884  0.298963  alpha 1,1
Jena     STAGED_P alpha    0.279884  0.298963  same as basic
Jena     STAGED_RAW        0.286285  0.302583  matched control
Jena     v8 JOINT_PQ       0.279714  0.299361  close prior control
Jena     DIRECT_NLINEAR    0.273984  0.286714  lower MSE
Jena     F0                0.292300  0.264758  higher MSE, lower MAE
Jena     LoRA              0.240590  0.226584  strongest here
```

Jena의 STAGED_P는 부모 대비 MSE를 0.044476 낮췄고 상대 변화는 -13.71%였다. paired block interval은 delta 기준 [-0.053305, -0.033137]로 음수에 머물렀다. STAGED_RAW 대비로도 delta -0.006401, 상대 -2.24%였고 interval은 [-0.007876, -0.005168]이었다. 그러나 v8 JOINT_PQ와의 차이는 0.000169로 작고, paired interval이 [-0.000805, 0.000908]라서 안정적 우위로 쓰면 안 된다. DIRECT_NLINEAR와 LoRA에는 각각 +2.15%, +16.33% 나쁘다.

기간별로도 같은 결론이다. Jena에서 STAGED_P는 TEST-A MSE 0.346512, TEST-B MSE 0.213255로 두 기간 모두 부모보다 낮았다. Robin에서는 TEST-A와 TEST-B 모두 부모와 같은 수치였다.

```text
dataset  role         TEST-A MSE/MAE      TEST-B MSE/MAE
Robin    LEVEL        0.273677 / 0.331865 0.276712 / 0.317319
Robin    STAGED_P     0.273677 / 0.331865 0.276712 / 0.317319
Robin    STAGED_RAW   0.275792 / 0.333415 0.277246 / 0.318619

Jena     LEVEL        0.422475 / 0.381041 0.226243 / 0.259859
Jena     STAGED_P     0.346512 / 0.344905 0.213255 / 0.253020
Jena     STAGED_RAW   0.354227 / 0.349115 0.218343 / 0.256052
```

Seed별 Jena STAGED_P는 92601 MSE 0.279772, 92602 MSE 0.279995로 서로 비슷했고, STAGED_RAW는 각각 0.286886, 0.285684였다. Robin STAGED_P는 부모 seed와 동일하게 92601 MSE 0.275009, 92602 MSE 0.275383이었다. Robin RAW는 92601이 부모와 같고 92602가 0.278032로 악화됐다.

## P/Q 진단

P/Q 분해는 complete-target subset 진단이다. masked macro MSE의 대체 지표가 아니며, 결측 target을 0으로 채운 투영이 아니다. Robin coverage는 complete rows 3,838/3,840, Jena coverage는 35,040/35,040이다.

Jena에서 STAGED_P는 P-space MSE를 부모 0.255840에서 0.211365로 낮췄고, Q-space MSE는 0.068519로 부모와 사실상 같았다. Q prediction parity도 두 seed 모두 pass였고, max absolute Q prediction difference는 0.000001 미만, violation은 0이었다. STAGED_RAW도 P-space MSE를 0.212916까지 낮췄지만 Q-space MSE가 0.073369로 증가했다. 따라서 Jena의 matched RAW 대비 이득은 “Q를 고정한 채 P를 보정한 절차”와 일관된다. 다만 이것은 노출된 Jena와 complete-target subset에서의 진단이며, 원인 증명이나 일반화 증거는 아니다.

Robin에서는 STAGED_P가 P와 Q 모두 부모와 같았다. 이것은 안정적 개선이 아니라 “보조 절차가 부모로 되돌아간 사례”다.

## 비용과 실용 판단

`cost_decision.json`은 Robin 비용 측정을 생략했다. 유용한 nonzero staged 후보가 남지 않았기 때문이다. Jena는 부모·matched RAW 대비 정확도 신호가 있어 같은 자료의29개 실행 구성,3개 순서 블록과 sentinel을 포함한99행을 측정했다. alpha1의 basic/auxiliary는 실제 부모·head tensor·함수 동일성을 확인해 같은 측정을 재사용했다. 독립 실행으로 중복 계상하지 않았다.

RTX4070, FP32/TF32off/CPU4threads, 공통 VAL24원점, 예열10·전체24원점20회 반복이다. 모델 로드·디스크·원장은 제외하고 표준화 CPU입력부터 온라인 부모+head, 전송, 전체 원채널 CPU출력까지 포함했다. 아래 중심값은 함수별3블록 중앙값을 다시 중앙값으로 취한 뒤 seed평균이다. reserved는 전체 관측 범위이며 allocated와 다른 지표다.

```text
Jena 방법          B1 ms   B4 원점/초   B4 allocated MiB   B4 reserved MiB
BASE_LEVEL         11.20     342.78          217.90             238
STAGED_P           11.64     325.52          217.97             238
STAGED_RAW         11.25     339.74          217.97             238
JOINT_PQ v8        11.30     324.90          217.97             238
F0 full            11.05     226.32          281.44             304
LoRA merged full   11.13     224.80          281.44             306
F0 B4 M24            —        90.99          216.89             252
LoRA B4 M24          —        88.45          216.89             252
DIRECT_NLINEAR      0.14   21249.73            9.56              22
```

P 보정은 full LoRA보다 낮은 할당 메모리와 높은 B4 처리량을 보였지만 MSE가16.33% 높다. LoRA도 행 microbatch24로 P보다 낮은 할당 메모리에 들어가며, 그 대신 처리량이 낮았다. P와 v8의324.90/325.52원점/초 차이는 안정적 속도 우위가 아니다. P의 seed별 B4 블록 중앙값 범위는 각각2.9885–3.1556 및2.9588–3.0736ms/원점이었다. F0 앞뒤 sentinel 변화는 -3.03%부터+1.64%였고 원순서·불리한 블록을 모두 보존했다. 원인을 특정한 시스템 병목 주장은 하지 않는다.

DIRECT_NLINEAR는 같은 GPU 세션에서 훨씬 빠르고 메모리가 작으며 MSE 점추정도 P보다 낮다. 대응 오차 구간은0을 포함하므로 정확도 우위를 확정하지 않지만, P의 실용 선택 이유를 약화하는 강한 대안이다. 작은 head 수나 C/K 비율만으로 배포를 권고하지 않는다. 다른 회차 timing과 비율을 만들지 않았다.

Jena P/RAW/v8의 전체 배포 parameter는47,754,108개, parameter bytes191,016,432, buffer 포함 tensor bytes191,016,468이다. 부모 LEVEL은47,736,188개/190,944,752 parameter bytes, F0와 병합 LoRA는47,718,016개/190,872,064bytes, DIRECT는24,624개/98,496bytes다.17,920은 v9의 증분 학습 변수이고 부모G까지 누적35,840개다. LoRA 적합 변수294,912개와 병합 후 전체 배포 크기도 구분했다. CPU 프로세스 RSS/VMS 등은 원행에 있고 GPU 프로세스 전체 점유는 이 측정의 별도 확인값이 없으므로 N/A다.

부모 epoch·과거 적합 시간, 새 head의 step/온라인 학습·VAL 시간, 보조 alpha 시간, 총 새 attempt 시간은 [training_summary.csv](training_summary.csv)에 별도로 있다. 새 target의 적응 비용에는 부모 적합도 필요하며 과거 부모 시간+이번 시간의 역사적 합산은 같은 세션의 통제된 학습 가속비가 아니다.

첫 비용 parity 실패는 모델 복원까지 감싼 inference_mode와 학습의 no_grad 경로 차이에서 관찰됐다. VAL-only 진단 후 모든 비용 비교군을 no_grad로 맞춰 고정 허용오차 그대로29개 정합성 검사를 통과한 뒤 측정했다. 원본 실패·코드·partial·진단은 [정정 기록](cost_checker_correction01.json)에 보존했고 학습·선택·TEST 값은 바꾸지 않았다.

실자료8/12attempt, 부모 새 적합0, 합성2/6세션(첫 실패0update·수정10update), GPU1131.394초/7200초로 GPU 작업을 마쳤다. CPU 검사와 저장 최종값은 닫힌 원장 및 게시 receipt를 따른다. 예비4fit는 사용하지 않았다. 검산 PASS는 실행팀의 정합성 확인이며 독립 재현이나 논문 기여 인증이 아니다.

## 다음 결정 한 가지

**이번 결정은 v9를 제한된 추가 보정·절제 결과로 유지하고, 기본 LEVEL 또는 주력 논문 방법의 전면 대체는 보류하는 것이다.** Jena의 부모/RAW 대비 발전은 실제 근거로 남는다. 동시에 Robin 추가 효능, v8 대비 우위, 선형/LoRA 대비 일반적인 선택 가치는 확보하지 못했다. TEST를 보고 자료별 routing을 만들지 않았다.

남은 결정 하나는 **논문의 중심 주장을 건물 전력에서 확인된 조건부 정확도–비용 절충으로 제한해 계속할 것인지**다. 범용 정확도 회복을 주장하려면 현재 근거로는 부족하다. 이번 회차에서 새 자료·모듈·튜닝을 추가하지 않고 결과와 반례를 원고 업데이트에 함께 남겼다.
