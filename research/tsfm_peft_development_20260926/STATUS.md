# 승인된 채널 압축 잔차 PEFT 개발

- 상태: INITIAL_COMPARISON_RUNNING (2026-09-26)
- 사용자 승인: 현재 대화의 전체 계획 승인 및 자율 실행 목표. PLAN.md는 직전 계획 응답 원문을 현재 대화 rollout에서 정확히 추출했다.
- 기준 commit: 06023468c36fb379554fd90f5d9b6947d5979af9, main, origin=CanelE452/hierarchical-tsfm-peft.
- 재개 확인: 승인 폴더/PLAN/STATUS가 없었으며 GPU Python 프로세스도 없었다. 완료된 신규 실험은 발견되지 않았다. 기존 미추적 history/transient 연구/체크포인트를 보존한다.
- 주력: 학습형 채널 압축 + 재구성 잔차 시간 예측. 예비 0개. 중심 가치는 점예측 정확도. 새 주제 확보는 아직 미확인.
- GPU 사용: 실시간 원장은 gpu_jobs.json (완료 smoke 약6초 + 현재 initial job elapsed_s). 신경망 fit attempts는 runs/*/attempt.json, 상한48. 구조·목적 수정0/2. 새 GPU는 루트 실행자만 사용한다.
- 현재 작업: 4군×2LR×2seed 최초16fit 순차 실행. exec session78200 / 실제 PID는 gpu_jobs.json 및 .cache/tsfm_peft_development_20260926/gpu.lock 확인. 프로세스가 살아있으면 재시작하지 않는다.
- 다음 행동: 완료된 curve/result에서 최적화 경계와 대조 효과 확인. 다음 GPU 평가는 동일 job 종료 후. 보호평가 접근금지 유지.
- 불확실성: 압축 잔차의 예측 유용성, pretrained backbone 추가 가치, 17 Bull 계열의 train 가용성/기존 노출, 실측 시간.
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
