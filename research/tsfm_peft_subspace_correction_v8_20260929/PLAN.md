# V8 승인된 개발 계획

아래 계획은 사용자가 전체 승인한 직전 응답의 실행 내용이다. 승인문은 APPROVAL.txt에 보존한다. v7 종료 main 0c9a08f22a9f01820859aa0e8648fc1601a21444에서 별도 회차로 시작한다.

**개선을 목표로 개발할 수 있습니다. 지금은 외부 재현을 늘리기보다, 실제로 손해를 만드는 구조적 제약을 시험하는 쪽이 맞습니다.** 다만 정확도 회복과 범용성 확보를 미리 보장할 수는 없습니다.

기존 코드와 정정된 진단을 다시 확인했고, 다음 개발안을 권합니다.

**1. 먼저 건드릴 문제**

[확인] 현재 LEVEL은 고정 PCA에서 **버린 공간의 오차만 보정하고, 보존한 공간의 TSFM 오차는 수정할 수 없습니다.** Robin의 LoRA 대비 손해는 두 공간에 모두 남아 있습니다. Jena에서는 원입력을 받는 RAW가 LEVEL보다 좋았습니다. 기존 근거는 ../tsfm_peft_practical_controls_v7_20260928/TOPIC_DECISION.md에 있습니다.

이는 보존 공간에도 보정 경로를 줄 근거입니다. 그러나 그 오차가 작은 선형 모델로 예측 가능한지는 아직 미확인입니다.

**2. 첫 변경: 두 공간을 독립적으로 보정**

P는 TRAIN PCA의 보존 공간, Q=I-P, delta X는 마지막 관측값을 뺀 입력, b=D(F0(E(X)))입니다.

제안식: Yhat=b+repeat48(x_last Q)+G_Q(delta XQ)+H_P(delta XP).

기존 잔차 보정 G_Q에 **보존 공간 보정 H_P** 하나를 추가합니다.

- 두 분기는 독립적인 공유 선형 512→32→48, 편향·활성화 없음.
- 출력층을 모두 0으로 시작해 초기 예측은 기존 LEVEL과 동일.
- F0·PCA·K는 고정. TSFM 처리 계열 수는 늘리지 않음.
- 학습 파라미터는 **17,920→35,840개**.
- 새 TRAIN 적합으로 비교하며 기존 학습 가중치를 유리한 초기값으로 복사하지 않음.

[제안·미검증] 이것은 **손해 일부에 접근할 수 있게 하는 변경**입니다. 기존 Q 분기의 표현력은 그대로이므로 모든 손해를 해결하는 설계는 아닙니다. K 증가도 가능하지만 TSFM 비용까지 늘어나므로 이번 첫 수정에서는 제외하겠습니다.

마지막값 제거·복원이나 잠재 어댑터 자체를 새 기여로 주장하지 않습니다. 해당 원리는 NLinear 저자 구현(https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py)과 AdaPTS(https://proceedings.mlr.press/v267/benechehab25a.html)에 연결됩니다.

**3. 파라미터만 늘려서 좋아졌는지 함께 확인**

대조식: Yhat_RAW64=b+repeat48(x_last Q)+G64(delta X).

G64=512→64→48로 **같은 35,840개 파라미터**를 줍니다. 두 모델의 초기 예측은 같지만 함수 공간까지 같다고 주장하지 않습니다.

```text
기본 실험                       fit
Robin: 수정 모델·RAW64 × 2seed    4
Jena:  수정 모델·RAW64 × 2seed    4
기술적 복구·재시도 여유           4
기본 8 / 최대 12 attempt
```

기존 LEVEL·RAW32·F0·LoRA·직접 NLinear 결과는 재사용합니다. 두 자료는 결과의 유불리와 관계없이 모두 수행합니다.

새 모델에는 같은 seed·TRAIN 순서와 기존 레시피를 적용합니다: LR 1e-3, AdamW, wd0, clip1, batch4, epoch당 512원점, 최대120epoch, strict patience6, 기존 plateau scheduler. step0 포함 최소 VAL checkpoint를 선택합니다. 실제 갱신·동결·복원·고정 TRAIN probe를 확인하고, 학습 부족을 방법의 실패로 단정하지 않습니다.

**4. 결과에 따른 결정**

- 수정 모델이 기존 LEVEL과 RAW64보다 유리하면, 공간별 독립 보정의 추가 근거로 유지합니다.
- RAW64로 충분하면, 복잡한 분리 구조를 채택할 이유가 약하다고 판단합니다.
- Robin만 좋아지고 Jena가 계속 나쁘면, **정확도 일부 회복과 도메인 확장 실패를 분리**해 보고합니다.
- 둘 다 정상 학습 후 불리하면 해당 수정을 기각합니다. 남은 예산으로 모듈을 계속 추가하지 않습니다.

MSE·MAE·기간·seed별 결과와 **F0/LoRA 대비 남은 손해**를 모두 보고합니다. 유지할 근거가 있을 때만 같은 회차 비용을 다시 측정합니다.

Robin과 Jena는 이미 노출됐으므로, 둘에서 개선돼도 곧바로 “범용성 해결”이라고 부르지 않습니다. 그때 고정한 방법으로 별도의 미사용 자료 확인이 필요합니다.

**5. 새 회차의 운영 상한**

[제안·미검증] 최대12attempt, GPU 1장·동시학습1개·GPU 점유2시간, 합성 검사6세션×최대10update, CPU 검사15분, 신규 저장5GiB로 제안합니다. 이전 회차의 남은 예산을 이월하는 것이 아닙니다.

승인 후 비충돌을 확인한 research/tsfm_peft_subspace_correction_v8_20260929/에 계획·원장·실행·판정을 남기고, 검산과 관련 변경의 origin/main 게시까지 진행합니다. 이 계획을 제시한 답변에서는 읽기만 했습니다.

전체 승인 질문: 이 v8 개발안 전체로 진행할까요?

승인 요청 근거: 새 구조가 기존 v7 승인 범위 밖이고, 적용한 plan-consensus 지침(C:/Users/User/.claude/skills/plan-consensus/SKILL.md)이 “합의 전 실행 금지”를 명시하기 때문입니다. 승인 후에는 위 범위의 구현·수리·실험·보고마다 다시 묻지 않습니다.

## 실행 구체화 — 승인된 범위 안의 고정 사항

- seed92601/92602; Robin C17/K5, Jena C21/K6; L512/H48. v7 TRAIN/VAL/TEST 배열·mask·원점·단위·PCA를 hash 검증하여 재사용한다. 원자료·백본 복제/다운로드0. 새 TEST/채널/기간 탐색 없음.
- 둘 모두 노출 개발자료이다. 이번 적합은 TRAIN만, checkpoint는 VAL만 사용한다. 두 자료의 8fit와 VAL 선택을 끝낸 뒤 신규 비교 TEST를 평가한다. 평가 후 추가적합 금지. 실패복구도 평가 이전에 수행한다. 240epoch 연장이나 새 LR 탐색은 승인 범위에 없다.
- Q head의 초기값은 v7의 동일 seed 초기 생성 순서를 재사용하고, P head는 별도 CPU fork_rng(seed+1000003)로 만든다. RAW64 down은 두 down의 행 연결, up은 0으로 한다. 두 모델 초기 출력, E/D 및 같은 seed/epoch의 표본 순서를 확인한다.
- 실제 데이터 optimizer 실행/실패/재시도는 모두 attempt. 실자료 smoke를 별도로 하지 않는다. 합성 optimizer 검사는 CPU에서 한 세션 전체 최대10update. GPU 부모 작업 점유에 포함된 fit 시간을 다시 더하지 않는다. root만 실행을 시작하고 모든 실행을 공통 원장에 먼저 예약한다.
- 검사는 초기·선택 가중치/gradient/갱신/고정 범위·mask·선택/재시작 복원에 한정한다. synthetic PASS를 실자료 학습 충분성으로 부르지 않는다.
- 주 MSE는 기존 observed-mask channel-macro MSE, 보조 MAE/seed/기간/channel; seed 손실 평균과 ensemble을 구분한다. P/Q 진단은 모든 비교군 공통 완전관측 target에서 별도로 수행한다. 결측 정답0채움 금지. 기존 v7에서 쓰던 period 내부 block 재표집(Robin7/Jena42원점,2000회)을 재사용할 수 있으며 새 합격선을 만들지 않는다.
- 비용은 유지할 근거가 남을 경우에 한해 Robin 새 pq_split/RAW64/기존 LEVEL/F0/merged LoRA를 같은 VAL24, FP32/TF32off/CPU4threads, B1/B4, CPU입력→전체CPU출력 범위에서 측정한다. 원형 실행을 주로 하고 미압축 B4 M20은 v7 실용 대조의 재측정으로 포함한다. warmup10,24원점20반복,3순서블록; block/seed 원자료 보존. Jena 비용은 측정하지 않으면 주장하지 않는다. 과거 timing과 비율 계산 금지.
- 새 시스템/전역환경/driver/power 변경·다른 프로젝트 종료·권한 우회 없음. 충돌 대기는 회당30분/누적60분을 넘기지 않는다. 승인 상한 자동증액 없음.
- 산출물: PLAN/APPROVAL/STATUS/protocol/원장, 실행별 result/curve/receipt, 선택/평가/진단, 조건부 비용, 비교 JSON/CSV와 필요한 그림, TOPIC_DECISION/검산/게시기록. 원래 v1–v7와 무관한 dirty 보존. raw/weights/prediction arrays는 local cache만. 관련 파일만 개별 stage·일반 commit·origin/main push·live SHA확인. 과학적 지지와 코드 통과/게시 완료는 구분한다.
