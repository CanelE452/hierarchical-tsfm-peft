# 실행 계약 보충: 사용자 gate 변경·자원 상한 해제

상태: **인간 승인 수신 / 새 비교 구현 및 실행 진행**. [검토한 원 비교안](../GATE_REVIEW_AND_MINIMAL_FOLLOWUP_KO_v1.md)과 과거 승인·결과·시계는 보존한다.

사용자 답변은 `call_e1682a01be79412a9ffe80d525932115:0`의 “게이트 수정 승인 — 기존 잔여 상한 내 실행”이다. 이어 사용자는 **“상한내수행하지 말고 쭉해줘”**라고 직접 지시했다. 이번 후속은 그 최신 지시에 따라 시간·누적 CPU/GPU·저장 상한을 이유로 중단하지 않는다. 과거 `AUTHORIZATION_v1.json`·`CAMPAIGN_MANIFEST_v1.json`의 상한·만료 시각을 수정하거나 사용량을 리셋하지 않는다. 새 승인 기록과 자원 receipts를 별도로 만들며 기존+신규 사용량을 구분한다.

자료·K·L/H·관측 mask·FIT/TUNE/개발 구간·주지표와 보호 TEST 접근 금지는 유지한다. 기존 job·결과·source·checkpoint를 보존하며 자동 commit/push·설치·대규모 backbone 학습은 하지 않는다. 실제 논문/PPT가 아닌 검산 가능한 한국어 실험 보고서를 만든다. 물리 메모리·실패 상태는 관찰하고 자산/구현 문제는 해당 경로의 기록으로 남긴다.

독립 검토에서 찾은 구현 계약을 다음처럼 명확히 한다.

- **PCA도 동일 FIT 공분산으로 whitening**하여 두 경로의 FIT 잠재 분산을1로 맞춘다. PCA는 상위 K분산 방향, ForeCA는 같은 whitening 공간에서 spectral entropy 방향을 선택한다. 같은 미래에의 적합성이나 F 불변성을 가정하지 않는다.
- **D는 원 관측 과거만**을 정답으로 사용한 채널별 masked affine 최소제곱 복원이다. causal FF의 비관측 값을 정답으로 세지 않는다. E/D의 FP64 적합·FP32 배포와 두 경로의 동일 클래스를 구분한다.
- **Native는 잠재 K채널 각각의512→48 affine head**다. 입력은 `z_past-z_last`, 출력은 `z_last+innovation`. E/D를 고정하고 그 뒤의 실제 C 미래에 대한 관측 channel-macro 정규화 MSE로 학습한다. AdamW .001·weight decay0·clip1·batch4·600 update, 두 seed·TUNE checkpoint 규칙을 고정한다. CP0의 head는0이며 seed는 배치 순서를 바꾼다.
- **새 공통 FP32 배포식:** `raw_last + ((latent_future - latent_last) @ decoder)`. `latent_last`는 `encode(history)[:,-1]`에서 가져온다. D bias는 과거 재구성에만 사용하고 forecast innovation에서는 상쇄한다. F/native/oracle 모두 같은 식이다. 이는 과거 MAIN의 연산 순서와 별도 계약이며 그 모델을 덮거나 같은 byte라고 주장하지 않는다.
- **실제 TUNE 정답은 정상 checkpoint 선택에 사용한다.** 선택에서 배제하는 것은 진단 oracle의 `E(Y_future)`와 복원 결과다. Oracle는 모든 C의 target이 실제 관측된 시점만 평가하고, 실제 방법도 동일 mask의 보조 지표를 만들어 비교한다. 부분 target의 임의 채움으로 oracle를 좋게 만들지 않는다. Oracle 결과를 배포 순위·학습 입력·분기·방법 선택에 사용하지 않는다.

PCA/ForeCA 결정적 적합4개·동일 클래스 D4개·Native 최대8조건·실제 F 추론·배포 비용 최대20조건/120반복·독립 검산과 보고를 끝까지 진행한다. COST의 첫5순서는 다섯 route의 circular 균형,6번째는 사전 seeded 순서이며 전체6회가 완전 균형이라고 주장하지 않는다. Raw·두 Bolt 경로·두 Native 경로를 같은 회차에 두고 oracle를 비용에서 제외한다. Native 비용은 seed92601이다.

실험의 성공·무차이·실패를 모두 남긴다. 이 한 묶음의 완료를 원래 연구 목표 해결로 바꾸지 않으며, 다음 판단은 원수치에서 도출한다.
