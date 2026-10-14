# Beijing 원자료 로더 수리와 순서 편차 — 재학습 전 봉인

2026-10-13. Beijing 수리 실행 전에 기록한다. 사용자 승인된 새 자료 실험을 같은 사전 방법으로 완료하기 위한 로더 수리다. 새로운 연구 방법·자료·가중치 탐색을 추가하지 않는다.

## 확인된 오류

원본 UCI ZIP SHA는 `b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4`다. 정상 `PRSA2017_Data_20130301-20170228.zip`와 무관한 최상위 `data.csv`, `test.csv`가 함께 있다. 초기 진단은 최상위 CSV만 확인해 정상 PRSA가 없다는 잘못된 결론을 냈다. root가 실제 ZIP member 목록을 확인한 뒤 중첩 ZIP을 포함한 정정 진단을 수행했다. 원 진단 파일도 보존했다.

정정 진단 `beijing_data_diagnosis/current_raw_metadata_v2.json`에서 Aotizhongxin member 1개, 35,064행, 2013-03-01 00:00부터 2017-02-28 23:00까지의 시간당 시각, 내부 중복0을 확인했다. 원 로더는 전체 member 이름 문자열에 Aotizhongxin이 있다는 이유로 station 컬럼이 없는 두 CSV도 붙였다. 정상35,064행+503행+503행=36,070행 중 시간 구성 컬럼이 없는1,006행이 NaT가 되어 중복1,005개로 검출됐다. 원자료의 Aotizhongxin 행을 삭제할 문제는 아니다.

## 수리 범위

- 원본 ZIP과 source SHA를 그대로 사용하고 새 URL/원자료 다운로드를 수행하지 않는다.
- 봉인된 `newdata.py`, `engine.py`, `alignment.py`, `new_run.py`, PLAN 및 기존 실패·성공 결과를 수정하지 않는다.
- `newdata_repair.py`의 별도 parser에서 member별 필수 schema·station·시각을 확인해 사전 고정 Aotizhongxin PRSA만 선택한다. 임의 중복 삭제·보간·새 station 선택은 하지 않는다.
- 채널 순서11개, FIT60/TUNE20/EVAL20, FIT256/TUNE64/EVAL48 stride, L512/H48, K=floor(C/4)=2, FIT-only 통계, 창마다 reset한 causal FF와 실제 미래 관측 마스크는 원 계약과 같다.
- 기존 METHOD_LOCK SHA `38fde1bf35b785684944c785a9b7bec7b12cec0f2608fc0c202de04d0632cc82`의 architecture `ANCHOR_NONLINEAR_D`, 두 seed,600 updates, 학습·checkpoint 선택 규칙 및 각 방법 weight를 그대로 사용한다.
- weight는 BASE_Q0, LATENT_FUTURE_MSE0.1, SUMMARY_MSE0.1, SHUFFLED_SUMMARY0.03, FIXED_REGIME0.03, LEARNED_REGIME0.1, SHUFFLED_REGIME0.03이다. Appliances EVAL 결과로 변경하지 않는다.
- 새 cache/실행 tag에서 Beijing22 core fits·4 native fits 및 같은 raw/ridge 기준군만 수행한다. 이미 성공한 legacy/Appliances fit·control은 재학습하지 않는다.

## 되돌릴 수 없는 평가 순서 편차

원 `run_new_v1/RESULT.json`은 `PARTIAL_FAILED_PATHS`이며 SHA `115dc6e7f5508e0b729acaf6118c906a83de311720a16fdc1d27321a44dba5e3`다. Beijing 자료 준비 실패 후 독립 가능한 Appliances 실험은 끝까지 진행해 해당 EVAL을 이미 처음 개봉했다. 따라서 원 계획의 “두 새 자료의 모델 선택을 모두 완료한 후 어떤 새 EVAL도 처음 개봉한다”는 공동 barrier는 충족되지 않았다. 수리로 이 과거 사실을 바꾸지 않는다.

Beijing은 이전 학습·checkpoint 선택·EVAL 평가가 모두0이다. 날짜·station·schema를 통한 오류 진단만 수행했다. 수리 후 Beijing 모델은 같은 사전 METHOD_LOCK과 FIT/TUNE로 적합·선택하고, 수리 source/모델/자료 SHA를 봉인한 별도 MODEL_SELECTION_LOCK 후 Beijing EVAL을 처음 연다. 두 자료의 결과를 최초 공동 barrier를 통과한 하나의 실험이라고 주장하지 않는다.

최종 resolved manifest는 원 PARTIAL 결과와 수리 결과의 SHA·시간·비용을 연결하고 `joint_models_before_any_new_EVAL=false`를 명시한다. 저장 배열·선택·동결·자료 계약의 검산 통과와 원 평가 순서 준수 여부를 별도로 보고한다. 보호된 기존 TEST 새 접근·대형 백본 갱신·자동 commit/push는 하지 않는다.
