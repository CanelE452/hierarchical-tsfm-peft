# 승인된 채널 압축 잔차 PEFT 개발

- 상태: REVISION1_RUNNING (2026-09-26)
- 사용자 승인: 현재 대화의 전체 계획 승인 및 자율 실행 목표. PLAN.md는 직전 계획 응답 원문을 현재 대화 rollout에서 정확히 추출했다.
- 기준 commit: 06023468c36fb379554fd90f5d9b6947d5979af9, main, origin=CanelE452/hierarchical-tsfm-peft.
- 재개 확인: 승인 폴더/PLAN/STATUS가 없었으며 GPU Python 프로세스도 없었다. 완료된 신규 실험은 발견되지 않았다. 기존 미추적 history/transient 연구/체크포인트를 보존한다.
- 주력: 학습형 채널 압축 + 재구성 잔차 시간 예측. 예비 0개. 중심 가치는 점예측 정확도. 새 주제 확보는 아직 미확인.
- GPU 사용: 현재 원장합계8770.36초(약2.44시간)/86400초. 22완료fit+실패1attempt+현재첫수정실행. 수정8fit완료시30완료/31attempt. 구조·목적수정1/2. 정확한실시간원장은gpu_jobs.json.
- 현재 작업: 초기16fit+연장6fit 및extended DEV평가 완료(실패1attempt 별도보존). 22완료/23attempt. rank8의원입력대비이득은남지만강한선형대조대비추가가치는여전히미확보. 첫구조수정rank32의동일용량RAW대조8fit가session34971/PID50376에서실행중이다.
- 다음 행동: revision1_rank32_specs.json의8fit(RESIDUAL/RAW×2LR×2seed),같은latent8·손실·초기화방식·effectivebatch4·정체종료규칙. 기존LoRA/COMPRESS는재사용하고revision1_rank32_cohort.json에서VAL-only선택후DEV비교. 이후해석가능한추가가치가남는지판단한다. Bull적응·보호평가는아직금지.
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

30. 연장 실행 시작: session99539/PID60552, run.py fits --specs extension_specs.json. 첫 LoRA 연장의 관측 epoch 시간은 약9초로 초기 약18초보다 짧다. extended_cohort.json은 기존조기종료10개+연장6개로16설정을 실행 중 미리 고정했다. 평가기는 결과id/설정/완전한seed짝을 확인하고 VAL 평균으로 LR을 선택한다. DEV 점수로 초기/연장 결과를 고르지 않는다.

31. 실행 전 CPU 진단 근거: 선택 RESIDUAL 두seed에서 고정PCA 잔차공간 오차가 factor-linear보다 높고, 학습된G는 rank8이 붕괴되지 않았다. fullTRAIN ridge residual weight의 top8 Frobenius energy는53.05%이지만 이 수치만으로 예측 용량 부족을 확정할 수 없다. 이를 구별하려고 동일TRAIN/PCA/penalty0.001/common map에서 residual map만 rank8/16/32/48 TRAIN 최적 reduced-rank ridge로 제한한 VAL 진단4개를 수행한다. 입력 covariance로 가중한 quadratic의 최적해를 사용하고 rank48 원본일치/인공 목적함수 검산을 포함한다. CPU fit성격의진단4개 추가(총22), 새DEV/Bull/GPU 접근 없음. 구조수정은 아직0/2.
32. 연장6fit 종료 후 검산→저장VAL분해→extended DEV평가를 session72349에 순차 연결했다. PID60552 종료 및6개완료/id일치가 모두 필요하며 어느 단계든 실패하면 다음으로 진행하지 않는다. cohort 선택의 seed누락반례를 포함한 CPU9검사 PASS.

33. 연장 중간 결과: LoRA고LR seed1은46epoch에서 정체 종료,선택40epoch VAL0.165126으로 초기0.165129와 거의 같다. RESIDUAL저LR seed1은63epoch에서 정체 종료,선택57epoch VAL0.183260으로 기존40epoch0.185112보다 개선됐지만 고LR0.178411에는 미달한다. 나머지4연장 및 VAL-only 군선택은 진행 중이며 부분결과로 DEV 선택을 바꾸지 않는다. 신규cache 사용량83.6MB/30GiB(조회시점).

34. 원본 중단: extended120_raw_bypass_92601_0.0001이 epoch30 검증 중 gpu_jobs.json.tmp→gpu_jobs.json 교체 WinError5로 실패했다. 시도/곡선/error와137.50초를 보존, 이 partial fit은 군선택·효과수치에 사용하지 않는다. 완료18fit은 영향 없음. 동일한 최종종료기록 저장은 성공했으므로 일시적 파일 점유 가능성이 있으나 점유 주체는 확인하지 못했다. 권한/ACL/다른프로세스는 변경하지 않았다. atomic save에 최대6회·누적1.55초 재시도를 적용하고 일시/영구 거부 반례에서 기존파일 보존과 오류전파를 확인했다(CPU11검사 PASS). 같은설정 retry1을 새attempt로 전체 재학습하고 미실행3개도 이어간다. 실패1개까지 계상하여 완료예정22fit/총23attempt, 구조수정0/2. 평가 cohort는 original→retry ID만 바꾼 extended_recovery_cohort.json이며 원본cohort도 보존한다.
35. CPU rank 진단 완료(8.18초): 동일 fullTRAIN/PCA8/common map에서 residual rank8/16/32/48의 VAL MSE는0.187223/0.178386/0.174407/0.174000이다. rank8 제한의손해0.013223는두기간에동일방향이며 rank32는fullrank에근접한다. weight replay0, rank48저장예측MSE차1.21e-11, 인공rank/목적함수검산PASS. 신경망 TSFM 경로와는최적화·함수류가달라 직접적인개선보장은없지만, 후속 잔차기의용량수정을검토할근거가강해졌다. CPU22fits/transform, DEV/Bull/GPU추가0.

36. 재시작 사전단계 실패: 원장에 저장된 한글 WinError 메시지를 Windows 기본cp949로 읽어 UnicodeDecodeError가 발생했다. GPUJob 진입 전이라 GPU/fit 추가0. run/evaluator/checks/verifier의JSON 읽기를 저장규칙과같은UTF-8로 명시하고, 한글 실패원장의 보존·재진입·lock해제를 실제GPU없는 lifecycle검사로 확인했다. CPU12검사 PASS. 이는원래파일교체문제에연결된직렬화버그이며방법실패/구조변경으로세지않는다.

37. 복구 학습은 session42065/PID38696에서 시작했고 실제epoch진행을확인했다. 후속session46590은해당PID종료→복구cohort16결과완료검사→CPU검산→저장VAL분해→extended DEV평가를순차수행한다. GPUJob lock/24시간원장/48attempt상한을그대로적용한다. 실패또는gate불충족시평가는진행하지않는다.

38. 저장수정의영향확인: 실패원본과retry1의공통30검증시점(epoch0–29)에서epoch/step/TRAINloss/VALMSE/VALMAE/LR이모두exact equal. retry는원래중단지점을정상통과했다. 파일오류만을이유로재실행했으며원본실패는유지한다.
39. 다음수정후보를미리구체화: 잔차기만rank8→32, latent8/E/D/backbone/손실/표본/optimizer는유지. 같은rank32 RAW와함께2LR×2seed=8fit를revision1_rank32_specs.json에준비했다. rank32는CPU진단에서fullrank와VAL차0.000407로근접했으며 rank16에는0.004386 차이가남았다. 이는신경망개선의증명이아니다. 현재연장/평가가끝난뒤TSFM추가가치가여전히미확보이고해석가능한병목이라판단되면첫구조수정으로실행한다. 현재는준비만했고실행cycle0/2. 이득이지지되면동일질문의불필요한추가실험을하지않는다.

40. 코드기반 주장경계 점검: G는bias없는공유시간선형함수라 RESIDUAL은 D(F(EX)-G(EX))+G(X)로도쓸수있다. 두군은같은원정보/파라미터수를쓰지만 잔차군에는E/D를통한추가gradient가있고, 학습후DE는직교projection이아니다. CPU진단은고정PCA·affine ridge·전체TRAIN최적화이므로신경망rank8의하한이아니다(실제neural VAL은CPU rank8보다낮음). FACTOR-LINEAR는강한실용대조이며사전학습유무만의인과절제는아니다. F0의좋은DEV성능은이과제에서의유의미한기준성능으로만해석한다. rank32는측정후보중fullrank에가까운첫설정이지보편적인충분rank가아니다. 이점검의동의자체는과학적증거로세지않는다.

41. 복구 진행: RAW저LR seed1은80epoch/선택74/VAL0.211069로정상종료했지만고LR0.202339보다높다. RESIDUAL저LR seed2도정체종료했고고LR보다높다. 아직마지막고LR잔차와RAW저LR seed2의연장및완료후DEV비교가남았다. 원본저학습량문제와방법추가가치부족을구별하여계속비교한다.

42. 첫구조수정의실행근거확정: RESIDUAL저LR는두seed모두정체종료(첫seed63/선택57,두번째88/선택82). 고LR두번째연장도46/선택40/VAL0.178316으로정체종료하여선택군평균VAL약0.178363는factor-linear0.174000보다높다. 최초seed고LR도37epoch정체종료여서현재선택군의부족을단순학습상한만으로설명하기어렵다. CPU rank제약진단과함께rank32만늘리는첫수정비교를수행할근거로삼는다. 기대되는지지는같은용량RAW보다잔차분리이득이남고강한선형대조와의차이가줄어드는것이다. 반대로RAW만따라잡거나선형대조로충분하면방법주장을낮춘다. 현재마지막연장/extended DEV평가종료후8fit를실행한다. 최종군선택은각군2seed평균VAL로하며기존LoRA/COMPRESS8fit를재사용하는cohort16설정을미리저장했다.
43. 보호계약검사준비: protected_protocol.py는봉인된Bull full spec과data_contract/NPZ/protocol/model/run 해시를검사한다. 합성seal CPU45검사PASS. 현재학습코드에는아직연결하지않았고실제final_seal도생성하지않았다. 보호평가를앞당기거나과학적근거로세지않는다.

44. 연장완료: RAW저LR seed2도70epoch/선택64/VAL0.215464로정체종료했다(정확수치는result.json). 모든6연장설정은최대120에도달하기전에정체종료했고,22완료fit 저장검산오차최대9.44e-16. extended DEV의RESIDUAL0.176131/RAW0.203863/factor-linear0.178457/F00.164537/LoRA0.161096. factor대비이득1.303%,두기간+3.046%/-0.726%,조건부block95[-1.816%,3.931%]. 학습량보완후에도추가가치불확실성이유지된다. 이근거와CPU용량진단에따라첫구조수정rank32비교8fit로진행한다. 별도후보추가/주장전환/보호평가없음.

45. 첫수정실행시작: session34971/PID50376,rank32 RESIDUAL/RAW총8fit. 구조수정1/2이며기존GPU/48attempt상한을유지한다. 후속session49747은해당PID종료와cohort16결과완료확인후검산→저장VAL분해→revision1_rank32 DEV평가로진행한다. 합성보호계약검사를포함한현재CPU57검사PASS. 실제Bull최종seal/적응/보호평가는여전히없다.

## 추가 선행 확인과 주장 경계

- [COSA 공식 논문집](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2a8ce71baac4c89bf9ff479d8240c7d9-Abstract-Conference.html): ICLR2026 게재 확인. 공식 초록과 [저자 README](https://github.com/bigbases/COSA_ICLR2026)를 읽었으며 전체 구현은 검증하지 않았다. 동결 예측기 출력과 최근 관측 통계로 선형·gate 잔차를 적응한다.
- [ORCA](https://arxiv.org/html/2606.14222v1): 2026-06-12 arXiv v1, 이 조회에서 학회 게재는 확인하지 않았다. 원문 §3.1–3.6을 읽었다. 입력·기존 예측에 조건화한 출력 오차를 선형 경로로 보정하며, 온라인 buffer·routing·predictive-space regularization을 사용한다.
- 판단: 작은 잔차 경로나 선형 보정 자체를 신규성으로 주장하지 않는다. 현재 후보는 학습형 채널 압축의 재구성 잔차 R=X-D(E(X))를 별도 예측한다는 선택으로 구별되지만, 식의 차이만으로 추가 가치가 증명되지는 않는다. RAW-BYPASS와 FACTOR-LINEAR를 넘는 근거가 필요하다. 이 확인으로 데이터 계약·방법·실행 범위를 확대하지 않았다.
