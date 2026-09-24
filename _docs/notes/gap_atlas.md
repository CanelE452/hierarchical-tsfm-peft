# 격차 지도 (gap atlas) — GIFT-Eval에서 관측된 격차의 재현·안정성·위치

계열 문서. Stage 1은 계획자가 리더보드에서 채굴했고, Stage 2가 이 저장소의 실행 대상이다.
계약: `experiments/gap_atlas_stage2_20260924/00_MASTER_CLI_gap_atlas_stage2_20260924.txt`
(sha256 앞 16자리 `c16578f0f7e74197`).

## 1. 제안 (2026-09-24 착수 시점)

가설: GIFT-Eval 리더보드에서 Chronos-2 zero-shot이 데이터별 학습 모델에 크게 지는 설정
(bitbrains_rnd 계열 등)이 실재하는 격차이고, 같은 데이터의 다른 기간에서도 유지된다.

방법: 후보 4개 + 음성 대조 3개 설정에서, 공식 test 창(P2)과 그 직전 같은 수의 창(P1) 두 기간을
각각 독립적으로 학습·선택·평가한다. arm은 F0(zero-shot) / LORA / FULL / DL(PatchTST·NHITS) /
TOTO(보고 전용) / SNAIVE(정상 동작 확인). 선택은 각 기간의 검증 블록으로만 한다.

판정 지표:
- E0a 공식 재현 — 7개 설정에서 F0 CRPS가 리더보드 chronos-2 값과 상대 2% 이내. 실패하면 STOP_DEBUG.
- E0b SNAIVE가 리더보드 seasonal_naive와 상대 5% 이내.
- S2a 음성 대조 3개 모두 두 기간에서 격차 G < 0.10. 하나라도 넘으면 ATLAS_CONTROL_FAIL.
- S2b 후보별 G_P1 ≥ 0.10 이고 G_P2 ≥ 0.10 이며 두 기간 CI 하한 > 0 → STABLE_GAP.

예상 실패 모드:
- 우리 LORA·FULL·DL이 리더보드 딥러닝보다 약해 격차를 재현하지 못함 → "격차 없음"이 아니라
  "참조가 약함". GAP_TABLE에 우리 DL과 리더보드 같은 모델 점수를 나란히 적어 구분한다.
- bitbrains 격차가 CRPS에만 있고 MASE에는 없음(Stage 1 관측) → 분포 보정 문제일 수 있고,
  그렇다면 PEFT 주제가 아니다. L6 포함률에서 드러나게 한다.
- 격차가 소수 계열의 극단값에 몰림 → L3에서 확인하고 가설을 그 계열 유형으로 좁힌다.

중단 기준: E0a 실패 즉시 STOP_DEBUG. 음성 대조 실패 시 ATLAS_CONTROL_FAIL로 종료.
총 wall-clock 10시간 초과 시 진행 중 단계만 마무리하고 TIMEBOX_EXCEEDED 병기.

## 2. 실행 환경 확인 (착수 전)

[확인] 계약이 지목한 `gift_eval` 로더·GIFT-Eval 데이터·딥러닝 라이브러리·Toto 가중치는 이 PC에 없었다.
- `gift_eval`(salesforce-gift-eval)은 `numpy~=1.26`, `datasets~=2.17`, `scipy~=1.11`을 고정한다.
  현재 환경은 numpy 2.4.6 / datasets 4.8.5 / scipy 1.17.1 / torch 2.10 이라, 패키지째 설치하면
  기존 계약(TCV-1 등)의 환경이 깨진다. 그래서 패키지 설치 대신 공식 로더 `src/gift_eval/data.py`를
  그대로 가져다 쓰고(vendoring), 실제로 필요한 `gluonts`·`toolz`·`python-dotenv`만 추가한다.
  셋 다 기존 패키지를 바꾸지 않는다([확인] pip dry-run).
- 공식 로더가 실제로 쓰는 것: `datasets.load_from_disk`, `gluonts`(ProcessDataEntry/split/Map/
  norm_freq_str/Transformation), `pandas`, `pyarrow`, `toolz`, `dotenv`.
- 필요한 데이터는 네 개(bitbrains_rnd, bitbrains_fast_storage, bizitobs_l2c, electricity)로 약 1.1 GiB.

## 3. 결과

(채워 나간다)
