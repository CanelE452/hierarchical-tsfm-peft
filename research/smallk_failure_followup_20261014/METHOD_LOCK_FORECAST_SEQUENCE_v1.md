# Forecast sequence residual decoder v1 — 실행 계약

상태는 **SOURCE_ONLY_NOT_EXECUTED**이다. 새 runner/test 소스는 작성했지만 실행·테스트하지 않았다. 초기 모듈의 기존 12 CPU toy tests 통과와 이번 새 runner 검증 상태를 혼동하지 않는다. 실제 실행은 사용자가 승인한 추가 자원 파일과 run manifest를 root가 생성하고, root-owned `continuation_runtime_v1.ContinuationLimit(tag,gpu)`가 승인 gate를 통과한 다음에만 가능하다. 원래 campaign wall/cap을 재설정하지 않는다.

## 고정 비교

Robin C17/K4, Beijing Aotizhongxin C11/K2; 각 old selected MAIN seed92601/92602에서 STATIC_RF1/TEMPORAL_RF3: 정확히 8개 신규 fit이다. 동일 dataset/seed의 양쪽은 같은 기존 E/D 체크포인트, FIT 통계와 masks/origins, causal E와 current-C anchor, frozen Chronos-Bolt-small F를 사용한다. old partial checkpoint나 joint/APP 신규 결과에서 이어 학습하지 않는다. 초기 old CP는 이미 actual-Y TUNE으로 선택되었다.

head와 경로는 고정 SHA f61bbeb1bdb104c420bbf46cd41d3b8b74c1d9931dc326739191b8303108853d인 `forecast_sequence_decoder.py`의 구현을 사용한다. STATIC은 GELU MLP width22/21, TEMPORAL은 causal Conv1d K→16 kernel3 후 GELU와 16→C kernel1이다. 새 residual은 predicted latent 경로의 R에서 같은 past-prefix와 last-latent 반복 경로의 R을 빼며, 실제 미래값·미래 mask·teacher·추가 raw-C 입력이 없다. 등록 수와 상쇄 bias/active 수 차이는 기존 `PLAN_FORECAST_SEQUENCE_DECODER_KO.md`에 공개한 그대로이다. 기존 MAIN 연산 순서는 보존한다.

새 head의 final weight/bias는 zero이며 all3 split의 초기 source-GPU 출력과 old MAIN cached output을 atol1e-5/rtol2e-6으로 비교한다. 새 residual은 정확히0이어야 한다. all3 warmstart 배열을 저장한다. 첫 head hidden gradient가 zero인 것은 초기화의 예상 결과이다. 이후 E/D/head의 actual update proof와 F-input branch gradient를 따로 검사한다. old sum/subtract anchor의 FP rounding을 persistence error에 포함해 보고한다.

## 데이터·학습·선택

동일 old bridge FIT256/TUNE64; Robin 기존 노출 VAL30, Beijing 기존 노출 EVAL146; L512/H48을 재사용한다. TEST 접근·새 독립 평가·기존 중단 campaign 재개·F weight 재학습·commit/push·논문/PPT 작성은 하지 않는다. 과거 이 subset을 여러 번 본 사실을 독립 일반화로 바꾸어 말하지 않는다.

E/D/head만 fresh AdamW lr.001, wd0, clip1로 600회 새 update한다. batch4이며 `default_rng(original_seed).integers(0,Nfit,(600,4))`를 양쪽이 공유한다. 학습은 actual-Y normalized observed-channel macro MSE이다. residual을 mask 후 제곱해 unobserved NaN backward를 피하고 zero-observed channel은 평균에서 제외한다. 관측 target nonfinite/empty batch는 보존된 실패로 끝낸다. 실제 원자료에 NaN이 없다면 robustness 수정이 실패 원인을 해결했다고 말하지 않는다.

CP0/100/200/400/600마다 FIT/TUNE prediction, absolute future latent, last2 past latent, base/head residual, targets/masks/origins/stats와 점수를 저장한다. 실제-Y TUNE MSE로 고르며 exact tie는 earliest, checkpoint0 포함이다. 평가 split은 선택에 사용하지 않는다. u600 학습 증거와 selected 상태를 별도 저장한다. selected0은 기존 예측 유지이지 신규 개선이 아니다.

## 산출물과 감사 계약

각 fit은 새 `.cache/smallk_failure_followup_20261014/forecast_sequence_v1/<tag>`만 생성한다. RESULT는 단일 조건 객체로 dataset/seed/head/objective='Y', CP lineage, bridge/source snapshots, same initialization key, order SHA, 모든600 curve, gradient diagnostics, learning_executed를 기록한다. `warmstart_all_splits`와 `selected_inference`는 hashed NPZ descriptor다. selected NPZ는 prediction/absolute_future_z/past_z_last_two/base_MAIN_style_prediction/head_residual/raw_last/target/observed/origins/mean/std/original_RAW_forecast/original_MAIN_forecast를 포함한다. 독립 CPU head/D 재구성이 가능하다.

새 CP는 codec/head state를 분리하고 dataset/seed/head_type/objective/new_update/architecture를 저장한다. descriptor의 `state_sha_old_algorithm`, `state_sha_Bi_algorithm`은 codec; `head_state_sha_old_algorithm`, `head_state_sha_Bi_algorithm`은 head; `combined_state_sha_Bi_algorithm`은 prefix를 붙인 전체 state이다. 파일 SHA·identity·각 state SHA를 검증한 뒤 선택 상태를 strict restore한다. F old/Bi before/after equality와 F parameter grad=None을 종료 시 재검사한다. original source/CP/RAW/MAIN file SHA는 재검증하고 덮지 않는다.

손실·점수는 frozen joint v2 SHA352cb68c499a84ce9e5e32b95f4adfb2af96c89b226da28eec44630a3b26d21f의 finite-safe observed loss, scoring, earliest-selection을 재사용한다. 원 자산/architecture loader는 frozen condition-probe SHA94874b92486a86f56eac46dccd3ec9fe59b161aeda109f98d2f84ff0fd7c7648을 재사용한다. 실제 execution 전에 SHA를 확인한다. 새 runtime source도 snapshot한다.

## 자원과 실패

각 fit process cap150초/RSS8GiB이다. root-approved continuation shared campaign/GPU/CPU/storage/RSS/VRAM cap을 base 매 training/predict batch에 검사한다. full check는 매10 update, CP 경계, 모든 CP/NPZ write 전후, source check 전후 수행한다. constructor는 asset read보다 앞선 승인 gate이다. receipt/terminal failure를 RESULT에 반영하고 status가 COMPLETE가 아니면 exit1이다. close 실패나 F 변경을 COMPLETE로 숨기지 않는다. 부분 curve/CP를 보존하고 자동 resume/cap 확대는 하지 않는다.

8 fit의 GPU process cap 합은1200초, 다음 paired 비용 측정의 계획 상한180초로 구조 비교는1380초다. 독립 audit CPU 계획 상한300초다. 추가 승인 GPU1800초/CPU1800초/wall3600초/storage2GiB가 실제 승인되면, root는 APP 미완료 비용·audit/report 비용을 먼저 공유 manifest에 배분하고 잔여 cap 내에서만 실행한다. 지금 이 파일은 승인 상태를 추정하지 않는다. importer/driver/tests도 실제 검증 전이며 runtime API `.check/.shared.check/.rt.BudgetHeld/.close` 계약은 root review에서 확정한다.

## 다음 판단의 한계

RF3가 유리할지 모른다는 것은 가설이다. 추가 head의 capacity 차이, E/D 동시 갱신, 2seed, 반복 노출 자료 때문에 실험 하나로 channel-lag 복원 실패의 원인을 확정할 수 없다. masked MSE/MAE, 채널별 오류/큰 오차, 비용을 함께 보고한다. 합성 lag toy의 함수 표현성과 실제 forecasting 개선은 분리한다. 실제 GPU F나 매 update forward를 독립 재실행하지 않은 CPU 감사는 그 범위를 명시한다.
