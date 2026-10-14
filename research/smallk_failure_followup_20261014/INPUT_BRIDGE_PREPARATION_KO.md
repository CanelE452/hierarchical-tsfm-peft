# 이전 조건 입력 브리지 준비 기록

상태: **PREPARATION_ONLY**. [확인: 소스 작성] `prepare_inputs.py`와 합성 CPU 테스트를 추가했다. 이번 준비에서는 이전 scientific NPZ, CP, 모델, 실제 target payload, GPU, 보호 TEST를 열거나 실행하지 않았다.

## 목적

이 브리지는 이후 D1/D2 진단이 이전 작은 K 캠페인의 입력 조건을 정확히 재사용하도록 입력 NPZ를 만드는 전용 단계다. 후속 판단은 “작은 K 경로가 나빴는가”가 아니라 “같은 RAW/MAIN/NATIVE 조건, 같은 origin/mask/stat/history에서 표현 손실과 예측 변환 문제가 어디에서 생기는가”다.

도메인은 시계열 forecasting이다. 누수 경로는 시간 split 혼합, VAL/EVAL 재사용을 독립 증거처럼 부르는 것, FIT 통계 재계산, EVAL target 재개봉, 보호 TEST 접근이다. 따라서 브리지는 다음을 강제한다.

- Bi 최종 `AUDIT_RESULT.json`의 `whole_experiment_complete=true`, `failed_count=0`, 모든 check 통과, 그리고 `MANIFEST.json`의 report/audit SHA 일치가 확인될 때만 `--execute-after-bi`가 실제 이전 파일 해시와 NPZ를 읽는다. 보고서 안의 보존된 과거 실패 기록 문자열은 차단 조건으로 쓰지 않는다.
- 보호 TEST 이름과 알려진 TEST 경로는 audit hook으로 차단한다.
- Robin은 원본 `research/past_only_repr_diagnosis_20261012/common.py`의 `common.load_data('robin', sizes=(256,64,None))` physical/mask/origin을 쓰되, 최종 mean/std는 canonical RAW prediction NPZ의 저장 FP32 통계로 override한다. archive copy `run_dev_v1/sources/common.py`는 같은 SHA provenance로만 기록한다.
- Robin history는 old `experiment.preprocess`의 per-window causal observed-only forward fill을 사용한다.
- Beijing은 기존 repair contract, FIT/TUNE payload, 저장 EVAL normalized history를 사용한다.
- Beijing은 `newdata_repair.open_evaluation`을 호출하지 않는다. EVAL target/mask는 이미 노출된 canonical RAW prediction NPZ에서만 가져온다.
- RAW, MAIN 92601/92602, NATIVE 92601/92602의 mean/std/target/mask/origin이 exact match인지 phase 이후 검증한다.

## 산출물

실제 실행 시 fresh follow-up cache 아래에만 쓴다.

```text
.cache/smallk_failure_followup_20261014/input_bridge/<fresh_name>/
  robin_fit_input.npz
  robin_tune_input.npz
  robin_val_input.npz
  beijing_aotizhongxin_fit_input.npz
  beijing_aotizhongxin_tune_input.npz
  beijing_aotizhongxin_eval_input.npz
  FOLLOWUP_RUNTIME_RECEIPT.json
  RESULT.json
```

각 split NPZ schema:

```text
history             float32 [N,512,C]  normalized exact old-condition past
history_physical    float32 [N,512,C]  old causal preprocess 전 raw physical past, missing은 NaN
physical_past       float32 [N,512,C]  history_physical alias
raw_forecast        float32 [N,48,C]   canonical RAW normalized forecast
target              float32 [N,48,C]   canonical RAW stored target
observed            bool    [N,48,C]
history_observed    bool    [N,512,C]
physical_past_mask  bool    [N,512,C]  history_observed와 exact match
origins             int64   [N]
raw_last            float32 [N,C]
mean, std           float32 [C]        canonical RAW stored stats
columns             string/object [C]
common_load_mean    float32 [C]        audit-only common.load_data/load_training stats
common_load_std     float32 [C]        audit-only common.load_data/load_training stats
```

`RESULT.json`에는 phase barrier refs, source/file SHA, per-array SHA, split roles, origin counts, reference equivalence, lineage, shared resource receipt, and actual-status flags를 저장한다. CP는 file SHA만 검증하고 `torch.load`하지 않는다. 모델 snapshot state hash는 기존 asset map 선언으로 연결하되 이 브리지가 모델을 로드하지 않는다.

자원 API는 `followup_runtime.FollowupLimit(tag, gpu)`다. 이 adapter는 생성 전까지 Bi runtime/torch를 import하지 않고, actual 생성 때 동결된 `research/bihash_future_prediction_20261013/runtime.py` SHA를 확인한 뒤 원 `runtime.Limit`을 만든다. 독립 audit body lowerbound `407.8757245s`와 SHA 확인된 report body v1 `49.6082627s`, v2 `51.25142159965s`를 합친 `508.73540879965s`를 추가 CPU charge로 한 번 반영하고, follow-up source/cache storage를 Bi storage와 합산해 12GiB cap을 검사한다. root verification RT CPU는 이미 shared ledger에 있으므로 중복 가산하지 않는다. ad-hoc startup/test overhead는 unknown으로 기록하고 0으로 가정하지 않는다.

## 실행 인터페이스

준비 상태 확인:

```powershell
& ./.venv/Scripts/python.exe research/smallk_failure_followup_20261014/prepare_inputs.py --describe
```

Bi 완료 뒤 root 실행:

```powershell
& ./.venv/Scripts/python.exe research/smallk_failure_followup_20261014/prepare_inputs.py `
  --execute-after-bi `
  --bi-audit <completed_bi_audit.json> `
  --bi-report <completed_bi_report.md> `
  --bi-manifest <completed_bi_report_manifest.json> `
  --output-dir .cache/smallk_failure_followup_20261014/input_bridge/<fresh_name>
```

`--output-dir`는 존재하지 않는 fresh 디렉터리여야 한다. phase barrier는 최종 audit/manifest SHA 계약만 본다. historic failed path 보존 기록은 history로 취급하며, 최종 `failed_count`와 check 결과가 권위 있다.

## 준비 검증

합성 테스트는 다음을 확인한다.

- stored RAW mean/std가 common stats보다 우선한다.
- old window preprocess가 per-origin reset과 missing mask를 보존한다.
- target/mask/origin은 canonical RAW payload에서 보존된다. reference 비교는 exact이며 matching NaN은 같은 저장 의미로 허용한다.
- `history_physical`/`physical_past`는 old causal preprocess 전 physical window와 mask를 유지한다.
- TEST split은 schema에서 거부된다.
- Bi phase barrier가 미완료 상태에서 차단된다.
- Beijing repair physical manifest/payload가 선존재하지 않으면 loader 호출 전에 차단된다.
- Beijing EVAL history는 저장 normalized input을 사용하고 `open_evaluation`은 호출하지 않는다.
- RAW/MAIN/NATIVE equivalence drift는 오류가 된다.
- output NPZ는 exclusive write이며 array SHA를 기록한다.
- shared `FollowupLimit` adapter를 사용하며 bridge-local private ledger로 allowance를 늘리지 않는다.

현재 문서는 실제 실행 완료 보고서가 아니다. 실제 old-condition input SHA와 bundle은 root가 canonical final v2 phase refs로 gated CLI를 실행한 뒤 생성한다.
