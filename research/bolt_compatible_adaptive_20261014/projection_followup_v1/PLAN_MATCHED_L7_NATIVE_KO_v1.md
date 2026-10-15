# 고정 L7 RECON E/D에서 Bolt와 단순 잠재 예측기 비교

사전 계획이며 새 native 학습·새 비용 측정은 아직0이다. 직전 E-only2조건의 과학 감사와 새 공간4계산은 PASS/COMPLETE이며 보고서 독립 검산을 진행 중이다. SOURCE/합성 준비는 병행하고 그 보고 묶음이 검산된 뒤 실제 실행한다. 최신 계속 실행 지시 안의 같은 연구 목적에 대한 작업 판단이며 새로운 개별 승인을 받았다고 쓰지 않는다.

## 목적과 이미 한 비교의 경계

[확인] 고정 RECON D의 E-only DEV actual MSE .492266798/.505250004는 Raw .340401887보다 높다. 동일 complete-C retained numeric minimum .306905621/.311177336은 Raw보다 낮고 E(Y)는 .344458315/.356808128이다. 원 PCA E/RECON D의 DEV actual .488093014, E(Y) .332944728, minimum .306885041도 기존 실제 JSON에 있다. 낮은 공간 minimum이나 실제 미래를 쓰는 E(Y)는 과거만으로 그 미래를 알아낼 수 있다는 증명이 아니다.

[확인] 기존 native 8/12fit은 static/inverse 또는 per-origin shrink K×C D였다. causal-lag24D 실험은 new_native_head_training=0이고 현재 7K×C L7 RECON D에 맞춘 native는 미실행이다. 기존 LATENT_FUTURE_MSE도 actual-C+고정 teacher 회귀 보조항으로 이미12fit600update 실행했으므로 예측 목적 표현/잠재 미래 회귀의 첫 시도라고 부르지 않는다. 이번 질문은 새 대조학습 설계가 아니라 **같은 고정 E/D·과거 입력에서 Bolt와 단순 예측기의 격차**다.

## 고정 조건과 새 6fit

Robin C17/K4, L512/H48, FIT256/TUNE64/기노출 VAL30, 원 origin/scaler/mask·현재 C anchor·lag [0,1,2,4,8,16,32]를 유지한다. E/D/center를 모두 고정한 세 상태를 쓴다: 원 RECON CP0, E-only seed92601의 선정 CP400, seed92602의 선정 CP600. D32는 모두 같은 원 FIT 과거 RECON 계수다. 후자 E는 FIT 미래 actual-C 감독과 TUNE 선정 이력이 있으며 Bolt를 통해 최적화된 상태이므로 native에 최적화됐다고 하지 않는다.

각 상태에서 기존 native_head_v1.NativeHead의 K별 독립 affine512→48 예측기를 두 minibatch-order seed92601/92602로 fresh 학습한다. 총6fit×600=3600update다. weight/bias는0이며 seed는 초기 가중치가 아니라 순서 차이다. latent history-last를 입력하고 latent last를 되더하는 기존 식을 유지한다. Native CP0의 잠재 미래는 persistence이며 L7 known-past prefix 때문에 최종 C가 current-C anchor와 같지는 않다. 새 forward는 원 LinearLagAdapter와 predict_components를 가능한 한 재사용해 current-C+lagged innovation@D32의 순서·dtype를 유지한다. static FrozenProjection에 7K×C D를 억지로 넣지 않는다.

head만 AdamW lr.001/WD0/clip1/B4로 학습한다. 새 기존-E 비교와 같은 고정 full-FIT 관측수 가중 actual-C macro 정규화 MSE를 쓴다. 과거 native의 minibatch 관측수 분모 손실과 동일하다고 쓰지 않는다. 입력은 history뿐이며 FIT 미래는 loss target, TUNE 미래는 CP 점수다. E/D/F를 새로 학습하지 않고 F는 native 학습에 불필요하다. 기존 Bolt의 세 상태 실제 추론은 검산된 캐시를 재사용하며 F를 새로 학습한 대조로 세지 않는다. 이는 같은 학습 계산량이나 전체 학습 이력을 맞춘 대조가 아니다.

CP/선정 평가의 표현 추출은 기존 Bolt와 같은 CUDA FP32 E·canonical32 배치로 수행한다. 학습은 기존 E-only와 같이 B4 raw history를 E로 encode하며 canonical32 평가 캐시를 학습 전체에 재사용하는 것이 아니다. native를 CPU encode한 뒤 byte-exact라고 주장하지 않는다. 원 E0의 FIT/DEV는 기존 linear_e_train_v1의 CP0_LS_and_RAW_baselines/robin/L7에서 실제 Torch latent_history와 latent_future를 재사용한다. 그 캐시의 C 예측은 FORECAST-LS D를 썼으므로 현재 RECON D의 주비교값으로 가져오지 않는다. 같은 초기 E/center를 확인하고 재사용 actual F latent를 현재 RECON D로 Torch decode해 새 baseline C artifact를 저장한다. TUNE는 기존 joint RECON CP0_TUNE의 실제 Torch component를 쓴다. 새 E0 baseline C decode3개와 F forward0을 구분하고 기존 NumPy RECON CP0는 원 공차의 참고 parity로 남긴다. E-only 선정 두 상태의 기존 canonical32 Bolt component는 그대로 대조한다. E0 baseline을 포함한 모든 DEV 배열 읽기/새 decode도6잠금 이후다.

원 NumPy RECON과 새 Torch E0 C 예측의 비교는 backend 문맥 검사로 수치·원 공차 통과 여부를 보존한다. 이 문맥 검사 실패가 유효한 같은-Torch 비교 경로를 자동 생략시키지는 않는다. 주 gate는 원 입력/상태 결속과 Torch cached history byte identity, native zero-head future byte identity, 같은 latent를 쓰는 독립 NumPy lag식 parity 및 E-only 선정 Bolt C bytes 보존이다. 문맥 실패를 PASS로 바꾸거나 주 gate의 공차를 완화하지 않는다.

CP0/100/200/400/600 전체 TUNE MSE 최소·이른 tie를 유지한다. 같은 head seed의600×4 순서를 기존 E-only 순서 SHA와 대조한다. 모든6개 성공/실패 잠금·registry 뒤에만 DEV 배열을 읽는다. 한 상태/seed 실패로 독립 경로를 생략하지 않는다. head30CP와 선정 FIT/DEV12예측, 원 TUNE CP 점수·학습 로그·실패를 보존한다.

## 기술 검사와 검산

fresh 모델의 E/D는 requires_grad=False/grad None, center 고정, optimizer는 head parameter ID만 포함한다. 매 step flag 기록과 CP/final E/D/center byte 불변을 구분하며 저장 CP만으로 중간 bytes를 독립 재생했다고 주장하지 않는다. head가 실제 갱신되고 finite인지 확인한다. CP0의 latent history는 같은 기존 Bolt 상태/입력과 byte-exact로, zero head의 미래는 latent-last와 byte-exact로 대조한다. C 예측은 원 NumPy lag식과 기존 등록 atol1e-5/rtol2e-6 안에서 대조하며 결과를 보고 공차를 바꾸지 않는다. 기존 E-only/RECON CP0 데이터·상태/코드 SHA를 보존한다.

독립 감사는 저장된 head의 NumPy affine 예측·E/lag/D 산술·원 mask/정규화·TUNE 선정·로그/순서/잠금·CP fixed state를 검산한다. 이는 F forward나 optimizer/매 update loss의 독립 재생이 아니다. 새 예측기에 대한 actual/EY/flat·현재 C persistence·같은 D/offset numeric minimum을 같은 complete-C 정의로 계산하고, 계산과 외부 검산 범위를 나눈다. 미래 oracle는 학습·선정·배포에 넣지 않는다. E(Y)/minimum은 같은 E/D별 진단이며 두 head seed의 독립 자료처럼 세지 않는다.

## 실제 전체 비용과 판단

첫 비용 대조에서는 native head와 E/D, Bolt를 모두 같은 CUDA backend에서 추론하고 전처리와 최종 물리 반환은 기존 CPU 경계를 유지한다. native의 CPU 최적 구현 비용이나 모든 backend의 최솟값을 측정했다고 하지 않는다.

실제 학습/과학 검산 뒤 Native6+Bolt3+Raw1의10경로를 같은 원 물리 TUNE 첫B 입력으로 B1/8에서 측정한다. 20셀, warmup2회 뒤 고정 순서 seed20261015의5순환100표본이다. 5순환은 각 route가 전체10위치를 완전 균형으로 차지한다는 뜻은 아니다. 물리 입력의 causal fill/FIT 정규화→실제 E/predictor/lag D→물리 CPU 반환/sync를 포함한다. hash/setup/적합/선정/warmup/저장/guard는 밖이다. 같은 cohort Raw와 같은 E의 Bolt를 분모로 쓴다. repeated origins와 공유 resident 메모리를 독립 자료/독점 메모리로 바꾸지 않는다. F 실제 비용 경로는 기존 동결 체크포인트를 다시 load해 추론할 수 있으나 대규모 파라미터 학습은 없다. 기존 whole-route callable·physical precheck는 재사용하되 native head weight/bias도 adapter E/D와 별도로 전후 hash를 검사한다.

native가 Bolt보다 낮은 DEV MSE여도 Raw보다 높으면 연구 목표 해결이 아니다. native가 나빠도 이 단순 독립 affine head의 실패이지 모든 예측기의 불가능성이나 Bolt 필수성의 증명이 아니다. 두 seed·세 상태 전부 원수치/MAE/물리 단위/비용/실패를 한국어 MD·CSV·그림에 남긴다. 이미 노출된 DEV를 보고 정한 적응적 탐색이며 독립 일반화 검증이 아니다. K streams+K 과거 buffer+현재 C anchor여서 strict K-only도 아니다.

종전 CPU/GPU/wall 상한을 종료 기준으로 삼지 않되 같은 조건의 무조건 연장으로 결과를 찾지 않는다. TEST·설치/다운로드·큰 백본 파라미터 학습·기존 작업 종료/중단 학습 재개·자동 commit/push·논문/PPT는 하지 않는다. 현재 묶음을 끝낸 뒤 원수치에 따라 다음 원인을 좁힌다.
