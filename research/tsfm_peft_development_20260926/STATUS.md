# 승인된 채널 압축 잔차 PEFT 개발

- 상태: INITIAL_COMPARISON_RUNNING (2026-09-26)
- 사용자 승인: 현재 대화의 전체 계획 승인 및 자율 실행 목표. PLAN.md는 직전 계획 응답 원문을 현재 대화 rollout에서 정확히 추출했다.
- 기준 commit: 06023468c36fb379554fd90f5d9b6947d5979af9, main, origin=CanelE452/hierarchical-tsfm-peft.
- 재개 확인: 승인 폴더/PLAN/STATUS가 없었으며 GPU Python 프로세스도 없었다. 완료된 신규 실험은 발견되지 않았다. 기존 미추적 history/transient 연구/체크포인트를 보존한다.
- 주력: 학습형 채널 압축 + 재구성 잔차 시간 예측. 예비 0개. 중심 가치는 점예측 정확도. 새 주제 확보는 아직 미확인.
- GPU 사용: 실시간 원장은 gpu_jobs.json (완료 smoke 2.58초 + 현재 initial job elapsed_s). 신경망 fit attempts는 runs/*/attempt.json, 상한48. 구조·목적 수정0/2. 새 GPU는 루트 실행자만 사용한다.
- 현재 작업: 4군×2LR×2seed 최초16fit 순차 실행. exec session78200 / 실제 PID는 gpu_jobs.json 및 .cache/tsfm_peft_development_20260926/gpu.lock 확인. 프로세스가 살아있으면 재시작하지 않는다.
- 다음 행동: 최초 16fit 종료 후 학습 상한·선택 기록을 확인하고, 검증에서 선택한 설정으로 개발 구간을 비교한다. Bull 적응 전 결측 손실 집계를 보정한다. 다음 GPU 작업은 현재 job 종료 후 시작한다.
- 불확실성: 압축 잔차의 예측 유용성, 사전학습 backbone의 추가 가치, 원입력 보정 대비 차이, 학습량의 충분성, 보호 구간으로의 전이. Bull은 TRAIN 규칙으로16채널을 확정했으나 사전학습 중복은 미확인이다.
- 보호: Bull E1/E2의 예측/손실 접근 금지. Electricity 마지막20%는 이미 노출된 개발 자료.
- 권한: 승인 범위의 버그 수정·재실행·학습 조정 가능. 다른 프로세스 종료, 과거 기록 수정, 새 데이터/서버/총예산 증액 불가.
- 산출물: raw/checkpoint/prediction은 프로젝트 .cache/tsfm_peft_development_20260926/ (ignored); 공개 코드/설정/집계/원장/hash는 이 폴더.

## 변경 이력

1. 초기 재개: 실제 파일 부재 확인. 계획 원문 22612 bytes 보존. 기존 실험 재실행 없음.
2. CPU 구조검사 PASS, 데이터 검증6개 PASS. 실체크포인트 GPU검사 PASS: 공식출력/zero-residual/LoRA초기/복원오차0, gradient와동결범위확인. smoke3updates는학습충분성증거가아님.
3. Electricity 26304행×32채널, TRAIN17853합법원점/VAL108/DEV218. 모든TRAIN채널fully-observed/nonconstant. Bull17봉인ID중Trevor는TRAIN상수라제외,16채널. 보호E1/E2각40원점; 지표미열람. 계약/노출감사생성.
4. 데이터감사보강후 active fit의NPZ SHA256과현재archive동일(65b3e7db…): 학습중데이터변경없음.
5. 최초LoRA실측epoch약18초(512원점,128updates), 전체fit길이는아직미확정. epoch1~5검증MSE0.169666→0.168706,판정미완료.
6. CPU선형대조9회fit완료(약2.9초): VAL선택factor-linear DEV MSE0.188899,shared-linear0.188940,seasonal0.461101. DEV는재사용개발자료. GPU군선택/비교전이므로방법판정없음.
7. 첫 LoRA(92601,1e-5)는18epoch/333.2초로 종료,epoch12선택 VAL MSE0.168536. 실제 checkpoint 재생오차0. 다음 LR fit과 나머지 비교군은 진행 중이다.
8. CPU 평가 검산4개 PASS. 평가 전 non-circular 7-origin moving block으로 고정, seed별 대응 효과도 보고. 결측 bootstrap 표본에서 채널 count가0이면 CI를 조용히 만들지 않고 오류로 표시한다. 보호 데이터 추가 hash receipt는 protected_data_seal.json. 사전 봉인된 계약/ID/분할에는 변경 없음.
9. 코드 검토에서 원점별 MSE 평균과 effective-batch 채널 macro의 차이를 발견했다. Electricity는 완전 관측이라 현재16fit에는 영향 없다. Bull 적응 전에 microbatch에 전체 batch 채널 count를 적용하도록 수정하고 인공 결측 gradient 검산을 수행해야 한다. 실행 중인 run.py/model.py는 수정하지 않는다. 이는 구현 정합성 수정이며 새 방법 구조 수정이 아니다.
10. 최초 LoRA 두 fit 완료. 1e-4는40epoch/약728초,선택epoch40 VAL MSE0.165129로 상한에서 여전히 개선 중이다. 최초 비교 후 LoRA와 관련 대조의 연장 필요성을 검토한다. 저장된 두 fit의 예측을 scalar 방식으로 재계산한 MSE 오차0, 최소VAL 선택/원점 schedule/checkpoint hash 검산 PASS(fit_verification.json). 이는 별도 독립 재현이 아니다.
11. LoRA 등록 파라미터294912 중 실제 갱신245760. 갱신0인 항목은 decoder self-attention q뿐이다. 설치된 chronos_bolt.py:428에서 decoder sequence길이1, 이 attention의softmax값1이라q가출력에영향을주지않는 구조로 설명된다. encoder/cross-attention q와v는 갱신됨. 비용 보고에서 등록 수와 실갱신 수를 구분한다.
12. COMPRESS(92601,1e-4) 완료:24epoch,선택18epoch,검증 MSE0.266497,복원오차0,encoder/decoder 모두 갱신. 다른 학습률과 잔차/원입력 보정이 미완료라 주제 판정에는 사용하지 않는다. 현재 별도 구조 수정 없음.
13. COMPRESS(92601,1e-3)도 완료:20epoch,선택14epoch,검증 MSE0.261520,복원오차0. LoRA/COMPRESS 첫seed의 총4fit 저장 예측 검산 통과(최대MSE오차8.4e-16이하). RESIDUAL 첫 설정 학습 중. 게시되는4fit은 최초16fit의 부분 결과이며 최종 비교가 아니다.
