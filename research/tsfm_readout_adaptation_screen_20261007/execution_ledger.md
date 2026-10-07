# 실행 원장 — PLAN.md

- 2026-10-08: 직접 사용자 `해줘`로 표시된 S0 calibration 상한을 승인받았다. authorization.json에 수치 상한을 기록했다.
- 2026-10-08: 직접 사용자 `이런걸로 멈추지 마`에 따라 같은 범위의 구현·검증을 계속한다. S2/S3/S4 권한은 확대하지 않는다.
- Ruling: 기존 checkout의 새 campaign 디렉터리만 사용한다 — 원본 .cache/미추적 자료와 선택 부모가 이 checkout에 있고 기존 작업 보존이 사용자 계약이다 — 잘못된 경로 선택의 위험은 쓰기 경로 제한·보호 파일 해시로 검사한다.
- Ruling: 실행 계획의 검증 절차를 유지하면서 독립 코드 구현/검토를 agent에 위임한다 — 현재 developer의 병렬 위임 지침을 우선한다 — 공용 인터페이스는 중앙 실행 전에 확인한다.
- Ruling: S0 합성 CPU 검증 비용을 별도 receipt로 남기고 중앙 S0 원장에 합산한다 — 모델/GPU 호출이 없는 검증도 CPU 비용이다 — 중복 합산을 피한다.
- Task 1: model_readout.py/test_model_readout.py 작성, 중앙 합성 검증5개 PASS. 새 모델 로드0, GPU 호출0, 8.826초 wall/2.906초 CPU. 실제 native 검증은 아직 실행 전이다.
- Task 2: runtime/실행 보호장치 구현 및 교차 검토 중. calibration은 disposable updates378 이하, scientific fits0, TEST 접근0으로 수행한다.
- 사전 교차 검토 수정: optimizer/RNG checkpoint를 CPU로 읽어 Adam step scalar의 CPU 위치를 유지한다. cached HEAD와 online HEAD는 원래 출력 block을 각1회 호출한다.
- Task 2: S0 세 자료 native 검증 PASS. calibration378 + 계측 수정 integrity6 = native technical384 updates, scientific0/TEST0. 누적 GPU308.608초, CPU-only9.192초, 측정 CPU process340.063초. 별도 구현 fixture는 CPU toy40 updates/2.795초 wall이며 실제 CPU time은 미측정이므로 native calibration과 구분한다.
- Ruling: 반복 원장 저장을 매tick에서 1초 이하 heartbeat로 줄인다 — OUTPUT cached step에도 약0.09초가 섞였고 guard CPU check 실측은10.387ms에서0.00785ms로 줄었다 — 시간·counter cap은 매tick 검사하고 resource max/종료/오류/peak 초기화 전 강제 검사를 유지한다. 원본 runtime_s0_reference.py와 원래20-update 측정은 보존한다.
- Ruling: 최신 직접 계속 지시를 기존 S1 최대24 fits의 실행 승인으로 적용한다 — 사용자가 예산 절차로 다시 멈추지 말라고 명시했다 — 수치 예산 s1_budget.json을 실행 전에 기록·공개하고 실제 cap을 초과하면 자원 중단한다. S2는 별도 승인이다.
- Task 3: S1 코드·선택·bootstrap·폐기용 cost copy 교차 검토 완료. 120epochs 조기정지 미가정 raw projection13,793초, margin2.088으로 GPU8시간/전체9시간/total disk3GiB/RAM·VRAM8GiB 상한 선언. cold Adam single-step 이상치를 보수적으로 유지하며 single-step을 정상20-step block으로 바꾸지 않는다.
- Task 4: 고정24 fits 완료, scientific66,432 updates. 선택 epoch/LR와 source/checkpoint receipts 전체 봉인 때 TEST counter0을 확인하고12개 선택 모델만 평가했다. seed 예측 ensemble 없이 seed 손실 평균을 사용했다.
- Task 5: 동일 회차 폐기용 cost copies에서2,160 technical updates, training108/deploy162 blocks, sentinel18쌍 완료. Adam moments 생성 뒤 memory 및 선택 checkpoint pre/post/current hash 불변을 검산했다. CPU→전체 백본→CPU 배포 경로와 cached/online 학습을 분리했다. 전체 GPU job wall2,322.440초, 종료코드0.
- Task 6: CPU 검산 PASS,138개 실제 파일 hash와 기록된 channel 충분통계를 검산했다. 보호 기존 파일76개 변경0, 기존 dirty1,889개 inventory hash 동일. 큰 특징/데이터/가중치의 전체 hash는 앞선 driver/evaluator의 봉인을 참조하며 보고 단계에서 다시 모델이나 prediction arrays를 실행·로드하지 않았다.
- 보고 검토 수정: OUTPUT combined MAE 악화가 세 자료 모두임을 명시했다. S2 권고는 봉인된 HEAD VAL·정상 학습·실측 비용을 주 근거로 수정하고 TEST 후 구체화된 사후 개발 결정으로 공개했다. 이 수정은 report_readout.py/보고 산출물에 한정하며 봉인된 실행 소스·선택·과학적 checkpoint를 바꾸지 않았다.
- S1 종료: MIXED_READOUT_RESULT. Jena MSE OUTPUT9.31%/HEAD16.79% 개선, Robin OUTPUT20.26% 악화, Robin HEAD/Peacock 작은 차이는 조건부 구간에0 포함. 정상 epoch0 선택7개, OOM/상한 종료0. 기존 LoRA 격차는 unmatched-parent 참조이며 내부 추가 이득·PCA/G 유지·진단/신규성의 증명이 아니다.
- S2 제안만 기록: 모든3자료의 같은 selected HEAD 부모에서 HEAD_CONTINUE/HEAD_PLUS_LORA12 fits, 최대184,320추가updates. 측정된 별도 경로를 합산한 raw fit+VAL 추정21,378.9초는 joint 비용 실측/승인 cap이 아니다. 처음 joint 검증264 technical updates/GPU900초/RAM·VRAM8GiB 후보를 별도 승인 대상으로 제시하며 S2/S3/S4를 실행하지 않았다.
