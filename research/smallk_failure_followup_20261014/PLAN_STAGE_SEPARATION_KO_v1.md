# 표현·잠재 예측기·디코더 분리 진단 — 다음 승인용 계획

상태: **SOURCE_PLAN_ONLY / 미실행**. 이 문서는 관련 Python source만 읽고 작성했다. 새 CP·데이터·모델 읽기, 테스트, 학습, 추론을 하지 않았다. 지금까지의 frozen 소스·결과·receipt는 수정하지 않는다. 현재 승인 campaign은 23:48:18 UTC에 종료하며, 아래 추가 범위와 자원 승인 전에는 실행하지 않는다. 기존 ledger의 시간·사용량을 재설정하지 않는다.

## 먼저 답할 질문

[확인] RF1/RF3 8 fit은 실제600회 학습, F 고정, 독립 검산을 완료했지만 두 데이터의 평균 MSE에서 기존 MAIN을 개선하지 못했다. 이 사실은 «temporal decoder가 일반적으로 불가능하다»거나 «E에 정보가 없다»는 증거가 아니다. 현재 E/D는 입력 재구성을 목적으로 학습한 AE가 아니라, **frozen F를 통과한 실제-Y 예측 손실로 함께 학습된 좌표계**이다. 따라서 F 출력 대신 E(실제 미래)를 넣었을 때 D가 잘 복원해야 한다는 보장도 없다.

다음 질문은 (1) 같은 E의 과거 표현에서 단순 예측기가 유용한 미래를 만들 수 있는가, (2) 같은 잠재 예측에 D만 바꾸면 결과가 바뀌는가, (3) 실제 미래가 주어진 진단 좌표에서도 복원에 문제가 남는가이다. 현재 자료만으로 연구 방향이나 실패 원인을 확정하지 않는다. 최신 결과는 `report_sequence_followup_v1/REPORT_KO.md`에 있다.

## 기존 검사의 재사용과 중복 방지

source로 확인한 기존 계약은 다음과 같다.

- `run_saved_diagnostics.py:verify_native_lineage`는 기존 NATIVE control의 encoder/decoder state가 동일 old MAIN CP임을 CPU state equality로 검증한다. 이 NATIVE를 새로 600회 재학습하지 않고 기존 결과·비용·선택 기록을 재사용한다.
- `controls.py:fit_native`의 NATIVE는 시간축 `Linear(512,48)`를 각 latent에 공유하는 NativeShared다. E/D는 고정하고 **decoded actual-Y loss**로 학습했다. 아래 새로운 joint latent ridge와 같은 모델·학습 목표가 아니다.
- `controls.py:fit_ridge`의 RAW_JOINT_RIDGE 및 BOLT_RAW_R는 **raw-C block features**를 이용한다. 같은 E만의 정보로 예측하는 아래 native ridge와 구분한다.
- `engine.py:oracle`은 고정 선형 D column-space projection이며, nonlinear residual D에는 `not_applicable`이다. E(실제 미래)를 통과시키는 검사가 아니다.
- D0의 `reconstruction.csv`는 채널·tail 오차 기여도의 합이 총 gap을 재구성하는 산술 검사다. 입력 복원 성능과 혼동하지 않는다.
- D2는 old E/F/decoder hidden을 고정하고 F의 실제 예측 latent에 대한 outgoing D refit을 이미 실행했다. 해당 Y-refit과 7alpha/TUNE 기록을 재사용한다.
- `run_condition_probes.py:extract_absolute_latents`는 동일 old MAIN에서 전체 `latent_past`, `z_future`, `z_last`, source GPU prediction을 저장한다. 후속 실행 시 CP·통계·origin·SHA 계약이 일치하면 이를 재사용하며, 없거나 불일치하면 해당 자산을 사용할 수 있다고 추정하지 않는다.

## 동일하게 고정할 4개 원조건

Robin C17/K4와 Beijing Aotizhongxin C11/K2, old selected MAIN seed92601/92602: 정확히4개 원조건이다. 최근 head-fit CP나 중단된 CP에서 이어 학습하지 않는다. 기존 CP의 E/D와 F를 모두 고정한다. L512/H48, 원 FIT mean/std, origin, mask, FIT256/TUNE64, 기존 노출 Robin VAL30/Beijing EVAL146을 유지한다. 평가로 alpha·feature·구조·폭·rank를 선택하지 않는다. 보호 TEST·새 평가 데이터·대규모 backbone 학습은 없다. 기존 반복 노출 때문에 독립 일반화 검증이라고 쓰지 않는다.

공통 anchor는 기존 normalized raw current state \(x_t\)다. 추가 raw-C history를 native predictor에 주지 않는다. 기존 current-C anchor를 유지하므로 strict K-only 전송 구조라고 부르지 않는다. 배포 입력은 과거 X, 저장된 FIT 통계, 고정된 모델 계수뿐이다.

## 미래 oracle의 분리된 데이터 계약

배포 경로는 \(z_{past}=E(X_{t-511:t})\), \(\hat z_F=F(z_{past})\), \(x_t+D(\hat z_F)-D(z_t)\)다. 미래 Y와 미래 mask를 함수 입력으로 받지 않는다.

진단 전용으로만 normalized 과거512점과 실제 미래48점을 이어 **560점**에 causal E를 적용해 마지막48개의 \(z^*_h\)를 만든다. 각 미래 시점의 unobserved 채널은 직전까지 알려진 값으로 carry-forward하며, 첫 미래값이 누락되면 기존 filled past last를 사용한다. 뒤 미래값에서 역으로 채우거나 미래 통계를 계산하지 않는다. 원 future target의 결측값을 실제 관측치처럼 encoder에 넣지 않는다. past mean/std는 바꾸지 않는다. full560점의 과거 prefix가 기존 E(past)와 일치하는지 고정 atol1e-5/rtol2e-6으로 확인한다. 나중 미래를 교란해 이전 encoder 출력이 바뀌지 않는지도 확인한다.

이 oracle는 시점별 causal E를 쓰더라도 **예측 시점 t 이후의 실제 측정값을 알고 있는 진단**이다. past-only 예측 성능 표·배포 비용·누수 검사를 통과한 모델 순위에 넣지 않는다. 미래 oracle의 expected dependency와 배포 함수의 미래 불변성 검사를 별개로 둔다. oracle 계산이 성공했다고 미래 누수가 없거나 forecasting이 가능하다고 선언하지 않는다.

관측이 부족하면 \(z^*\)는 실제 미래와 causal imputation이 섞인 표현이다. 원 Y-observed mask 점수와 «h까지 미래 모든 C가 관측된 prefix»의 별도 점수/count를 함께 보고한다. native latent teacher의 primary 학습 mask는 이 complete-future-prefix 조건을 만족하는 h만 허용하고 K에 동일하게 적용한다. 해당 h의 FIT teacher count가0이면 그 h는 latent persistence(innovation0)를 유지하고 미학습으로 명시한다. teacher coverage가 부족하면 latent-native 해석은 보류하며, 직접-C native 경로는 원 관측 Y만으로 독립 진행한다. 결과를 보고 결측 기준을 완화하지 않는다.

## 고정할 신규 비교와 재사용 대조군

공통 past feature \(\phi(z_{past})\)는 기존 block_features와 같은 32개16-point mean 및 마지막 latent를 펼친 **33K**다. 두 새로운 predictor 모두 동일 feature와 FIT-only feature centering/scaling을 쓴다. zero-scale feature는 scale1이다. intercept는 unpenalized로 두며, 모든 coefficient·전처리 계수와 observed count를 저장한다.

N1의 complete-prefix latent teacher mask와 N2의 채널별 Y mask는 학습 가능한 표본 수가 다를 수 있다. 이 차이까지 공개하고, 두 predictor의 차이를 순수 decoder 효과라고 해석하지 않는다. N1↔N3의 고정-predictor 비교가 이를 보완한다.

각 원조건에서 다음을 수행한다.

1. **O0: true-future-encoder + old D.** \(x_t+D(z^*)-D(z_t)\). 학습 없는 진단이다. 실패는 E의 정보 손실 증명이 아니다. D가 기대하는 F 출력과 true-future E 좌표의 분포 불일치일 수 있다.
2. **O1: true-future-encoder + oracle-calibrated D.** 기존 D의 hidden32 GELU(Az+b)는 고정하고 outgoing linear weights만 FIT의 \([z^*-z_t,\ GELU(Az^*+b)-GELU(Az_t+b)]\)→observed Y−x_t로 ridge refit한다. 기존 D와 구조/hidden/anchor가 같고 FIT-only calibration이다. 이 D와 TUNE oracle 점수는 진단 전용이며 배포 후보의 선택에 가져오지 않는다.
3. **N1: joint latent ridge + old D.** 과거 \(\phi\)에서 FIT true-future latent innovation \(z^*_h-z_t\)를 학습해 \(\hat z_N=z_t+P\phi\)를 만든다. 위 teacher mask를 쓴다. 선택은 원 old D로 decode한 **actual-Y TUNE normalized MSE**이며 latent TUNE MSE로 대신하지 않는다. inference에서는 future/oracle 입력을 받지 않는다. 기존 NATIVE_SHARED actual-Y 학습과 달리 latent target을 쓰므로 목표 차이도 결과 해석에 남긴다.
4. **N2: joint direct-C ridge + current-C anchor.** 같은 과거 \(\phi\)에서 원 FIT observed normalized Y−x_t를 직접 학습한다. F와 D를 거치지 않는다. same-E past representation의 미래 readout 유용성을 검사한다. N1과 target/output dimension 및 계수 수가 다르므로 N2 우위만으로 D가 원인이라고 단정하지 않는다. Robin 계수 수는 intercept 포함 N1 약25,536/N2 약108,528, Beijing 약6,432/35,376이다.
5. **N3: 동일 N1 prediction + native-calibrated D.** N1에서 선택한 predictor 계수와 latent 예측을 그대로 고정한다. 같은 hidden32/outgoing D 구조를 **N1의 FIT past-only 예측 latent**→observed Y로 refit한다. TUNE도 N1의 past-only 예측 latent를 사용한다. O1의 oracle 선택 계수를 가져오지 않는다. N1↔N3는 predictor/E/input을 고정한 decoder 변경이며, predictor와 decoder를 동시에 바꾼 개선으로 보고하지 않는다.

existing F+oldD, 기존 D2 F+Y-refitD, 기존 SAME_MAIN_NATIVE, RAW 및 raw-C ridge는 재사용 대조군이다. N1의 alpha는 oldD 기준으로 먼저 선택해 고정한다. N3에서 N1 alpha를 다시 고르지 않는다. N2/O1/N3의 alpha는 각자 명시된 actual-Y TUNE 경로로 선택한다. 모든 후보와 선택하지 않은 점수도 남긴다.

고정 alpha는 **0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10**의7개, exact tie는 목록의 이른 값이다. FIT masked mean squared error와 scaled-weight ridge penalty를 사용한다. alpha0은 masked least-squares minimum-norm 해를 쓰고 conditioning/rank를 공개한다. decoder calibration은 D2와 같은 no-intercept FIT RMS feature rule을 유지해 latent innovation0에서 anchor를 보존한다. target residual은 관측 mask 후 손실/solver에 넣는다. zero-observed decoder channel은 원 outgoing weight를 유지하고 그 채널을 새 학습 성과로 세지 않는다. 새 ridge predictor의 미관측 output은 innovation0이며 따로 표시한다.

## 해석 gate

O0가 나쁘고 O1이 좋으면 decoder의 oracle 좌표 calibration 문제를 의심할 수 있다. oracle를 잘 복원해도 그 latent가 과거에서 예측 가능하다는 뜻은 아니다. O1도 나쁘면 기존 hidden feature/readout·결측·정규화·좌표 정보의 제약이 후보로 남으며, E의 보편적 불가능을 입증하지 않는다.

같은 E+oldD에서 N1이 F보다 좋으면 현재 F-path의 latent forecast/호환성 문제가 더 유력해진다. 학습 목표·predictor capacity가 다른 비교이므로 F 자체의 일반적 한계로 바꾸어 말하지 않는다. N2만 좋으면 same-E past feature에 유용한 정보가 있을 수 있지만 N1의 latent teacher/decoder/출력 차원 교란을 분리해야 한다. N1↔N3의 고정-predictor 차이가 decoder contribution의 더 직접적인 증거다. existing D2 F↔refitD와 비교해 predictor별 decoder calibration을 구분한다.

모두 좋지 않으면 이 작은 feature/readout·자료·선택 조건에서 개선 근거가 없다는 결과다. representation, 과거 예측 가능성, 미래 imputation과 distribution shift, 제한된 sample/alpha를 함께 남긴다. 실패를 해결했다고 쓰거나 큰 graph/contrastive learning이 필수라고 결론 내리지 않는다.

## 비용·저장·실행 승인

추가 승인 제안 상한은 **GPU process10분(600초), CPU process15분(900초), campaign wall40분, 저장 증가1GiB**다. 실제 소요는 미확정이다. 재사용 가능 여부를 먼저 검증하고, GPU는 oracle E 추출/엄격 parity/필요한 paired inference 비용, CPU는4원조건×신규4종×7alpha의 최대112 closed-form candidates·독립 검산·보고에 배분한다. backbone/encoder/기존 D hidden의 gradient 학습은 없다. native/decoder ridge coefficient 추정은 신규 학습으로 명확히 비용을 측정한다. 배포 시간은 oracle 미래 입력 없이 원 RAW/MAIN과 batch1/8을 같은 조건에서 paired 측정한다.

각 부분에서 CP·codec/E/D/F before/after hash, source snapshot, origin/stat/mask/coverage, past feature, diagnostic oracle future/imputation, alpha 전체 FIT/TUNE, selected coefficient, 실제 배포 prediction, normalized MSE/MAE·채널 count·물리 단위 채널 오차를 새 경로에 저장한다. oracle와 deployment 파일/표/함수명을 분리한다. 기존 D2/NATIVE/RAW 재사용은 실제 artifact SHA로 연결하고 없는 시간은 unknown으로 남긴다. 독립 CPU 검산은 저장 배열/계수/선택/score를 재계산하며 F-forward 독립 재실행으로 확대하지 않는다.

이 계획은 새 실행 승인이 아니다. 승인 뒤 root가 별도 authorization/manifest와 상한을 고정하고, 이전 만료 ledger/receipt와 결과를 보존해 추가 사용량만 기록해야 한다. gate 실패·자료 부족·자원 상한은 해당 경로의 실제 상태와 재개 조건으로 남기며 다른 독립 경로를 자동 생략하지 않는다. source-only 구현·fixture·실제 자산 확인을 마친 뒤 실행하고, cap을 자동 늘리거나 TEST/commit/push를 수행하지 않는다.

**현재 미확인 사항:** 실제4CP/D2-cache 파일 존재·array/codec SHA와 재사용 가능성, oracle 구성에 필요한 실제 future target/mask dtype 및 결측값 계약, 560점 real-device prefix parity, native/decoder 새 runner와 저장 schema·계산 시간은 이번 source-only 작업에서 확인하지 않았다. 실제 존재·검사 통과·완료로 적지 않는다. 추가 승인이 나더라도 만료된 v1 authorization/manifest를 덮거나 clock을 재설정하지 않고 별도의 새 승인 범위를 연결해야 한다.
