# Fixed-Q learning-equivalence audit

[확인] 기존 TRAIN/VAL만 사용했다. TEST 점수나 mask는 분기 입력이 아니다. 실제 optimizer update와 신규 fit은 0이다.

완전관측·동일가중 squared loss와 정확한 고정 직교 P/Q에서는 Q 유무 손실 차이가 LoRA에 대한 상수다. 실제 masked macro에서는 gradient 차이가 2 J_b^T W q이며, 채널별 관측수와 mask가 이 직교성을 깨뜨릴 수 있다.

제한된 실제 paired backward 24개를 수행했다. 최대 24개이며 미존재 결측 batch를 합성해 실자료처럼 보충하지 않았다. 각 pair는 동일 forward graph의 with-Q/main 손실을 두 번 미분했으며 가중치를 갱신하지 않았다.

허용오차는 FP32 elementwise atol 1e-6 + rtol 1e-4 × max(|g_full|, |g_main|)로 실행 전에 고정했다. 작은 차이나 통과는 전체 학습 궤적 동등성 증명이 아니다.

- robin: TRAIN 결측 target occurrence 48, VAL 0; paired backward 8; 최대 gradient 절대차 1.6406644e-05. 동등성 판정은 전체 궤적에 확대하지 않는다.
- jena: TRAIN 결측 target occurrence 624, VAL 20; paired backward 8; 최대 gradient 절대차 1.8144958e-05. 동등성 판정은 전체 궤적에 확대하지 않는다.
- hog: TRAIN 결측 target occurrence 48, VAL 0; paired backward 8; 최대 gradient 절대차 4.1571911e-06. 동등성 판정은 전체 궤적에 확대하지 않는다.

설치된 ReduceLROnPlateau min/rel 규칙과 v11 호출 순서를 scalar replay하여 기존 12개 A curve의 실제 LR과 대조했다. FP64 작은 선형 예제는 완전관측 일치와 비등방 mask 반례를 구분한다. 상대 threshold scheduler에는 상수 offset 반례가 존재한다.

완전관측 VAL에서는 동일 부모의 Q forward와 실제 VAL 정답으로 계산한 조건부 상수항도 기록했다. 이 항으로 기존 A loss sequence를 이동한 replay는 같은 A 궤적에서의 반사실적 scheduler 점검이다. 독립 no-Q 적합의 실제 궤적·최선 epoch·성능을 재구성한 결과가 아니다. 결측 VAL에서는 이 상수 이동을 실제 no-Q sequence로 제공하지 않는다.

기존 세 자료는 E1의 **same-checkpoint 출력 기여로 주장을 제한하는 분기**를 선택한다. 별도 NO_Q_DIRECT_TRAIN은 이번 좁은 출력 기여 판단에 필수적이지 않아 적합하지 않는다. 이는 수학적 중복 판정이 아니다. 독립 no-Q 학습·선택 정책을 이겼다는 주장과 전체 정책 동등성은 미확인으로 남긴다. 새 확인 단위의 TRAIN/VAL 계약은 별도로 검토해야 한다.

수치 원본: `equivalence_audit01.json`, `equivalence_mask_policy01.json`, `equivalence_paired_backward_ledger.json`, `equivalence_fixed_val_offsets01.json`, `equivalence_scheduler_replay01.json`, `NO_Q_RUN_DECISION.json`. 자체 검사이며 독립 재현이 아니다.
