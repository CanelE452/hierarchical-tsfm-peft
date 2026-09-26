# 승인된 채널 압축 잔차 PEFT 개발

- 상태: INITIAL_EVALUATED_EXTENSION_READY (2026-09-26)
- 사용자 승인: 현재 대화의 전체 계획 승인 및 자율 실행 목표. PLAN.md는 직전 계획 응답 원문을 현재 대화 rollout에서 정확히 추출했다.
- 기준 commit: 06023468c36fb379554fd90f5d9b6947d5979af9, main, origin=CanelE452/hierarchical-tsfm-peft.
- 재개 확인: 승인 폴더/PLAN/STATUS가 없었으며 GPU Python 프로세스도 없었다. 완료된 신규 실험은 발견되지 않았다. 기존 미추적 history/transient 연구/체크포인트를 보존한다.
- 주력: 학습형 채널 압축 + 재구성 잔차 시간 예측. 예비 0개. 중심 가치는 점예측 정확도. 새 주제 확보는 아직 미확인.
- GPU 사용: 현재 완료6706.82초(약1.86시간)/86400초, 정확한 실시간 원장은 gpu_jobs.json. 신경망16/48attempt 완료, 다음 연장6개를 포함하면22/48. 구조·목적 수정0/2. GPU는 루트 실행자만 사용한다.
- 현재 작업: 최초16fit·후속평가·비용 재검사·batch gradient 검사를 모두 종료했다. session78200/37265/38334/83891은 종료코드0. 초기 방법의 압축/원입력 대비 이득은 있으나 TSFM 추가 가치와 학습 충분성은 미확보다.
- 다음 행동: extension_specs.json의6개 설정을 최대120epoch, 동일 초기화/seed/LR/표본순서/effective batch4로 별도 fit한다. microbatch4는 실제 gradient 정합성 확인 후 사용한다. 기존 조기종료10fit는 재사용하고, 상한에 걸린6개를 연장 결과로 대체해 VAL-only 선택을 다시 한 뒤 같은DEV를 개발 비교한다. Bull 적응·보호 평가는 아직 금지한다.
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
14. RESIDUAL(92601,1e-4) 완료:40epoch,선택40epoch,검증 MSE0.185112. 상한에서 개선 중이라 학습량 충분성을 확정하지 않는다. 완료5fit 저장 예측/선택/checkpoint/표본 검산 통과. 두 번째 학습률 및 RAW-BYPASS·두 번째 seed 비교가 남아 있다.
15. 결측 MSE 수정식 CPU 검산: 원점별 loss를 평균하는 대신 effective batch 전체의 채널별 count로 각 microbatch squared sum을 나누어 더하면, full-batch loss와 gradient가 일치한다. 결측 반례는 기존13.375 vs 정확한50.5; 완전 관측에서는 동일하다. test_evaluation.py 총6개 PASS. 아직 실행 중인 학습 코드에 적용하지 않았으며, 최초 비교 종료 후 이 수식을 통합하고 통합 검산해야 한다.
16. RESIDUAL(92601,1e-3)는37epoch 조기 종료,선택31epoch,검증 MSE0.178411. 완료6fit의 저장 예측 검산 통과. RAW-BYPASS 첫 설정이 진행 중이며, 이 핵심 대조와 두 번째seed가 끝나기 전까지 잔차 분리의 추가 가치 판단을 유보한다.
17. 검증 두 기간의 MSE/MAE 보고를 평가기에 추가했다. 저장된 RESIDUAL(92601,1e-3) 검증 예측으로 전체0.178411 및 두 기간0.160062/0.196761의 집계 일치를 CPU 검산했다. checkpoint 선택은 기존 전체VAL 최저값 규칙을 유지한다.
18. 실행 전 진단 기록: 기존 CPU 선형 대조는512 TRAIN 창만 사용하지만 신경망은 매epoch512창을 다시 추출한다. 자료 활용량 차이가 TSFM 추가 가치를 과장할 수 있다는 우려를 검사하기 위해, 같은 모델·PCA·penalty3개·VAL선택으로 TRAIN17853창 전체를 쓰는 선형 대조를 추가한다. 아직 원인으로 확정하지 않는다. CPU 충분통계 적합9회 추가(총18회; 최초6–12는예상), thread2/chunk128. 데이터/분할·GPU/신경망 상한 변경 없음. 원본 결과는 보존하고 새 linear_fulltrain_baselines.json에 기록한다. 이 대조로 차이가 사라지면 TSFM 기여 주장을 낮춘다.
19. 강화 선형 대조 완료: 전체TRAIN17853창, CPU10.44초. VAL선택 penalty0.001에서 factor-linear VAL0.174000/DEV0.178457, shared-linear VAL0.177003/DEV0.180795. 현재 첫seed RESIDUAL의VAL0.178411보다 factor-linear가 낮아 TSFM 추가 가치는 아직 확보되지 않았다. 이후 주 비교는 이 강화 대조를 사용하며 최초512창 대조를 덮어쓰지 않는다. 충분통계와 직접 ridge의 인공 예측차이8e-15이하, 선택된4개 저장예측의 별도 scalar MSE 재계산 오차3.3e-11이하(저장float32 반올림 포함). 검산은 독립 재현이 아니다.
20. 첫seed8fit 완료. RAW-BYPASS(1e-4)는40epoch/선택40/VAL0.219982로 상한에서 개선 중이고, 1e-3은29epoch/선택23/VAL0.202339로 정체 종료했다. 완료8fit의 저장예측·선택·복원 기록 검산 PASS. 두 번째seed 실행 중. 현재 총GPU 점유 약1.03시간/24시간(정확한 누적은 실시간 원장), 구조 수정0/2. 미완료 비교와 학습 상한을 고려하여 주제 판단은 유보한다.
21. 저장된Electricity VAL 예측을 TRAIN PCA8 고정공간으로 분해했다(saved_validation_diagnostics.json). 첫seed RESIDUAL(1e-3)은 common0.079262/residual0.099149, 강화factor-linear는0.083186/0.090814. 전체 차이+0.004411은 직교잔차 쪽 손실+0.008335와 공통 쪽 이득-0.003924의 합이다. 앞기간 총MSE는0.160062 vs0.151135, 뒤기간은0.196761 vs0.196866. 이는 모델 내부 branch의 인과 분해가 아니며, backbone 기여나 저랭크 용량 부족을 확정하지 않는다. 전체·두기간의 분해/직교성/저장지표 일치 검산 통과, 새fit/모델예측/GPU/보호자료 접근0. 두 번째seed와 최초DEV 비교 후 학습량·잔차 용량 진단 필요성을 판단한다.
22. LoRA 두 번째seed도 완료하여 최초10/16fit 완료. 1e-5는11epoch/선택5/VAL0.168752, 1e-4는21epoch/선택15/VAL0.165801. 두 설정 모두 정체 규칙으로 종료했고 첫seed 고LR의 상한 개선과 구별한다. 완료10fit 저장결과 검산 PASS, CPU 분해는10신경망+2선형대조로 갱신. 현재 COMPRESS 두 번째seed 학습 중이며 핵심 잔차·원입력 대조가 남았다. 구조 수정0/2.
23. 최초12/16fit 완료, 두 번째seed RESIDUAL 학습 중. 최초 개발 평가기에 동일배치 평가의 wall time과 peak allocated GPU memory 측정을 추가했다. 입력/출력 전송·CPU지표·원장기록을 포함하고 모델 로딩은 제외하는 단회 측정이며 반복 latency benchmark로 부르지 않는다. 추가 예측 실행 없이 예정된 평가에 부착한다. 기존 fit별 peak 값은 각 fit 범위이며, GPUJob 원장의 peak는 fit 중 통계 reset 때문에 전체 job 최고치로 해석하면 안 된다. 최초job 종료 후 원장 표기를 명확히 한다.
24. 최초13/16fit 완료. 두 번째seed COMPRESS의 VAL은1e-4:0.267349(16epoch/선택10),1e-3:0.261644(13epoch/선택7). RESIDUAL1e-4는40epoch/선택40/VAL0.186650으로 두seed 모두 상한에서 개선 중이다. 학습 충분성을 미확정으로 남기며 최초 비교 후 관련 대조와 연장을 검토한다. 완료13fit 검산 및 저장VAL 오차분해 갱신 PASS. 현재 RESIDUAL1e-3 실행 중이고 RAW-BYPASS2fit가 남아 있다. 별도GPU평가·구조수정·보호평가 없음.
25. 후속 순차 실행을 session37265에 연결했다. 원래 PID59544가 살아 있는 동안30초 단위로 기다리고, 종료 후 initial_specs16개 전부의complete/id 일치가 있어야 CPU검산→오차분해→최초DEV평가로 진행한다. 앞 단계 실패 시 중단하며 기존 실패/결과를 덮어쓰지 않는다. 평가기는 VAL 두seed 평균으로 군별LR을 먼저 저장하고 Electricity DEV만 평가한다. GPUJob lock·총24시간 검사를 그대로 적용한다. 원래session78200과 후속session37265 모두 끝나기 전 run.py/model.py를 편집하거나 별도GPU실행을 시작하지 않는다.
26. 최초16fit와 자동DEV평가 완료. 두seed 평균MSE: RESIDUAL0.176130,COMPRESS0.259022,RAW-BYPASS0.203863,factor-linear0.178457,shared-linear0.180795,F00.164537,LoRA0.161096. RESIDUAL의 이득은 COMPRESS 대비32.00%,RAW 대비13.60%,factor 대비1.30%. factor 대비 두기간은+3.05%/-0.73%,조건부7일블록95%구간[-1.82%,3.94%]로 추가 가치가 불확실하다. 미압축F0/LoRA보다 각각7.05%/9.33% 높은MSE이며 이를 숨기지 않는다. 개발 재사용·선택 불확실성은 이 구간에 포함되지 않는다. 아직 주제 미확보.
27. 메모리 측정 오류 확인: 실제 LoRA 삭제 후192052736bytes가 남고 gc.collect 후0으로 해제됨(object_lifetime_audit.json). 최초fit별/평가별 allocator peak는 다른 모델 잔존분이 섞일 수 있어 모델간 비용 비교에 사용하지 않는다. 원본 수치 보존, 원장에 범위 주석을 추가했다. run/evaluator에 명시적GC와 job 전체 peak 추적을 보강하고 initial_cost_recheck로 동일선택·동일예측을 재평가했다. 저장10NPZ의 모든배열·선택·효과수치가 원본과 정확히 같음. 교정된평가 peak는 LoRA 두seed346351616bytes,RAW/RESIDUAL 두seed237306880bytes로 일치. 정확도 결과 무효화는 필요하지 않으며 비용에는 교정기록을 사용한다.
28. 결측 손실 보정 완료: run.backward_batch가 effective batch 전체 채널count로 누적한다. 완전관측/결측×micro1/4의 실제 학습함수 CPU 검사 포함8개 PASS. 실제4군×2mask의 GPU gradient 비교도PASS(최대상대L2오차5.84e-7,optimizer update0). 동일effective batch4를 한 번에 처리한 짧은 warmed backward 측정은 어댑터약0.10→0.026초,LoRA약0.13→0.057초; 학술 latency benchmark는 아님. 관측peak최대0.93GB로 로컬메모리 내이며 후속fit에 micro4를 명시한다. 모델 구조·목적 변경은 아니다.
29. 다음 실행 전 근거: 초기상한40에서 best epoch40인6설정(LoRA고LR seed1,RESIDUAL저LR 두seed·고LR seed2,RAW저LR 두seed)은 학습 충분성이 미확정이다. 초기원본을 보존하고 최대120epoch로 같은초기화부터 새fit한다. 기존조기종료·LR scheduler규칙은 유지하며120도 충분성 보장이 아니다. 재학습되는초기40epoch를 포함한 모든GPU시간과6attempt를 예산에 계상한다. microbatch변경은 동일목적·effective batch의 수치정합성 검사를 통과했지만 bitwise동일 궤적을 주장하지 않는다. RAW가 따라잡거나 선형대조 우위가 남으면 방법주장을 낮춘다. 근거 없는 모듈 추가는 하지 않는다.

## 추가 선행 확인과 주장 경계

- [COSA 공식 논문집](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2a8ce71baac4c89bf9ff479d8240c7d9-Abstract-Conference.html): ICLR2026 게재 확인. 공식 초록과 [저자 README](https://github.com/bigbases/COSA_ICLR2026)를 읽었으며 전체 구현은 검증하지 않았다. 동결 예측기 출력과 최근 관측 통계로 선형·gate 잔차를 적응한다.
- [ORCA](https://arxiv.org/html/2606.14222v1): 2026-06-12 arXiv v1, 이 조회에서 학회 게재는 확인하지 않았다. 원문 §3.1–3.6을 읽었다. 입력·기존 예측에 조건화한 출력 오차를 선형 경로로 보정하며, 온라인 buffer·routing·predictive-space regularization을 사용한다.
- 판단: 작은 잔차 경로나 선형 보정 자체를 신규성으로 주장하지 않는다. 현재 후보는 학습형 채널 압축의 재구성 잔차 R=X-D(E(X))를 별도 예측한다는 선택으로 구별되지만, 식의 차이만으로 추가 가치가 증명되지는 않는다. RAW-BYPASS와 FACTOR-LINEAR를 넘는 근거가 필요하다. 이 확인으로 데이터 계약·방법·실행 범위를 확대하지 않았다.
