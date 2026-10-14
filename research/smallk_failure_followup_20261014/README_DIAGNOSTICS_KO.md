# D0/D1 저장 예측 진단 코드

상태: **PREPARATION_ONLY — CPU synthetic 검사만 수행했다.**

[확인] 이 파일을 작성한 시점에는 기존 실제 scientific prediction NPZ를 열어 진단하거나 새 데이터셋·GPU·학습·모델 추론을 실행하지 않았다. 현재 Bi-Hash 연구가 끝난 뒤 root가 지시한 저장 자산으로 실행할 도구다. 원 실험과 결과를 덮어쓰지 않으며, 코드 실행 결과도 `COMPLETE_SAVED_OUTPUT_DIAGNOSTICS`로 표현한다. 이 상태는 후속 실험 완료가 아니다.

범위 근거는 [CONTINUATION_SCOPE_KO.md](CONTINUATION_SCOPE_KO.md)다. [diagnose_saved_outputs.py](diagnose_saved_outputs.py)는 CPU만 사용하며 [test_diagnose_saved_outputs.py](test_diagnose_saved_outputs.py)는 임시 폴더의 작은 synthetic 배열만 만든다.

## 비교 조건과 입력 계약

한 RAW 저장 NPZ와 같은 dataset의 후보별·seed별 NPZ를 받는다. 직접 경로 또는 source manifest 중 한 가지를 선택한다. 저장 값의 단위는 추정하지 않고 `normalized` 또는 `physical`을 명시한다.

기존 `smallk_predictive_alignment_20261013/engine.py`의 `add_prediction` 흐름은 각 split에 다음 배열과 공통 `mean`, `std`를 저장한다.

```text
mean, std                    [C], 저장한 FIT 정규화 통계
fit_prediction               [N,H,C]
fit_target                   [N,H,C]
fit_observed                 [N,H,C], bool
fit_origins                  [N], integer
tune_* / val_* / eval_*       요청한 split의 동일 필드
```

[확인] 준비 과정에서는 소스 코드와 기존 CSV/schema를 확인했으며 실제 scientific 배열의 내용을 읽지는 않았다. 코드에서 RAW·후보의 mean/std·target·observed·origin은 dtype·shape·값이 같아야 한다. `allclose`로 허용하지 않는다. 같은 위치의 NaN은 같게 취급한다. prediction은 동일 shape이고 모든 값이 유한해야 하며 실제 target은 관측 mask 위치에서 유한해야 한다.

관측값이 전혀 없는 채널은 정규화 지표의 활성 채널 수에서 제외한다. 부분 관측은 채널별 관측 수로 나눈다. std는 이미 저장된 양수의 effective FIT scale이어야 한다. 원 FIT가 상수여서 기존 데이터 계약에서 std를 1로 처리한 경우 그 값을 그대로 쓴다. std=0을 발견하면 거부하며 임의로 재적합하거나 1로 바꾸지 않는다.

FIT 통계라는 선언은 source manifest와 기존 원 실험의 데이터 검증에 의존한다. 이 도구는 원자료 FIT를 다시 읽어 mean/std의 학습 경계를 독립적으로 검증하지 않는다. 저장 origin과 mask가 같다는 사실만으로 과거 입력의 유효성이나 미래 누출 부재가 입증되지는 않는다.

split 역할은 `training`, `model_selection`, `reused_development`, `exposed_evaluation` 중 명시한다. `fit=training`이 필수다. eval은 `exposed_evaluation`일 때만 열 수 있으며 그 역할을 NPZ 읽기 전에 검사한다. `independent_evaluation` 선언은 이 도구에서 받지 않는다. protected TEST의 명시 경로를 거부하고 `test` split도 지원하지 않는다. origin이 중복되거나 단조 증가하지 않는 경우 비교를 거부한다.

## D0 출력과 해석

```text
파일                              내용
summary.csv                       seed별 normalized channel-macro MSE/MAE/bias와 RAW gap
channels.csv                      채널별 normalized·원단위 MSE/MAE/bias/absolute error 분위수
                                  관측 target/prediction mean·std, 저장 FIT mean/std
partitions.csv                    horizon별·1–8/9–24/25–H 구간·origin bin별 지표와 기여도
tails.csv                         RAW 고정 cutoff와 방법별 own cutoff의 tail/body 기여도
reconstruction.csv                각 분해의 합이 전체 MSE gap을 복원하는지 실제 수치
*_seed_metric_means.csv            각 seed 지표의 산술 평균, seed 수와 목록
REPORT_KO.md / RESULT.json         실행 범위·출처 SHA·split 역할·겹침·artifact SHA
```

정규화 주지표는 `C_active`개의 관측 채널에 대해 균등한 평균이다. normalized 저장 값은 그대로 사용하고 physical 저장 값은 명시된 FIT mean/std로 정규화한다. 원단위 MSE/MAE는 채널별로 보고한다. 단위가 다른 채널의 원단위 MSE를 하나로 합쳐 주지표와 비교하지 않는다.

저장 값을 float64로 변환해 재채점하므로 기존 실험의 FP32 단위 변환·reduction 기록과 작은 수치 차이가 생길 수 있다. 실제 실행에서 차이를 발견하면 재채점 수치와 원 기록을 구분해 보존하며 동일 결과라고 강제로 맞추지 않는다.

```text
MSE = (1 / C_active) * sum_c [ sum_(observed in c) e² / n_c ]

subset의 channel c MSE contribution
    = sum_(subset and observed in c) e² / n_c / C_active
```

분해 기여도의 분모 `n_c`와 `C_active`는 원 split 전체로 고정한다. 특정 horizon/time 부분의 단독 score는 그 부분에서 관측되는 채널들을 평균하므로 분해 기여도와 같은 수치가 아닐 수 있다. `contribution`은 더했을 때 원 split의 MSE와 gap을 정확히 복원한다.

origin bin은 저장 origin 순서에서 창 수를 최대한 같게 나눈 구간이다. 실제 날짜를 추정하지 않는다. 중복 origin을 거부해도 가까운 창의 미래 구간은 겹칠 수 있으므로 within/cross split의 target interval overlap pair 수를 함께 저장한다. 겹침이 없다는 사실도 독립 평가나 선택 이력의 봉인을 증명하지 않는다.

tail은 각 채널의 normalized absolute error 분위수로 정의하고 경계에 동률이 있으면 모두 포함한다. 따라서 q=.9가 항상 정확히 관측값의 10%를 고르는 것은 아니다.

- `same_split_raw`: 해당 split의 RAW 오류로 cutoff를 정하고 후보와 RAW에 같은 사건을 적용한다. 같은 사건의 tail gap이다.
- `fit_raw`: FIT의 RAW 오류로 cutoff를 정하고 다른 split에도 고정 적용한다. FIT에서 해당 채널 관측이 없으면 cutoff가 없으며 그 split의 채널을 body로 둔다.
- `same_split_own`: 각 방법의 자체 오류로 각각 cutoff와 사건을 정한다. 방법별 tail 규모를 보여주며 동일 사건의 paired 비교라고 해석하지 않는다.

same-split cutoff는 이미 저장된 평가 오류의 서술적 분해다. 이를 보고 선택한 채널·cutoff·개선 방법을 새 독립 검증에서 사전에 정한 것처럼 주장하면 안 된다. 모든 채널을 출력하며 큰 gap의 채널만 선택해 전체 성능이라고 쓰지 않는다.

seed 평균은 seed별 지표의 산술 평균이다. seed별 prediction을 평균한 ensemble 성능이 아니다. RAW 공통 참조는 한 번만 저장한다. seed 수를 독립 데이터셋 수나 독립 표본 수로 해석하지 않고 신뢰구간이나 유의성을 계산하지 않는다.

## D1 준비 범위와 남은 실행

선형 projection helper는 normalized `delta = RAW_forecast - current_last`를 사용한다. FIT delta의 모든 예측 채널·horizon에서만 basis를 적합한다. 실제 미래 target과 관측 target mask는 basis 적합에 사용하지 않는다. RAW forecast는 모든 채널에서 유한한 값이 있으므로 RAW 근사오차는 전체 C에 대해 채점하고 실제 target 예측오차는 관측 채널만 채점한다. 두 수치의 분모와 의미가 다르다.

```text
through_origin
    FIT second moment에서 rank K basis B 선택
    delta_hat = (delta B) Bᵀ

mean_centered
    FIT mean μ 고정, FIT centered covariance에서 rank K basis B 선택
    delta_hat = μ + ((delta - μ) B) Bᵀ

projected forecast = current_last + delta_hat
```

`rank_projection.csv`는 FIT의 retained energy fraction과 각 split의 RAW forecast 근사오차, 실제 Y 예측오차를 별도 열로 출력한다. centered 분산 설명률과 through-origin second moment 설명률은 서로 다른 수치다. FIT 에너지가 0이면 fraction을 정의하지 않고 effective rank 0을 명시한다. `rank_projection.npz`에는 고정 basis·mean과 split별 투영 예측을 저장한다.

이 선형 진단은 현재 학습된 nonlinear D와 같지 않고 nonlinear D의 보편적 오차 하한도 아니다. RAW full-C 추론을 먼저 요구하므로 빠른 경량 배포 방법의 성능으로 계산하지 않는다. 실제 미래 Y를 직접 투영하는 oracle은 구현하지 않았다.

현재 **`F(projected past)` 새 실제 추론과 `project(RAW forecast)`의 변환 순서 비교는 미구현·미실행**이다. root가 Bi 연구를 완료한 다음 과거 입력·last의 출처와 모델 조건을 고정하고 별도 경로로 수행해야 한다. 이 도구는 D1 중 저장 RAW forecast의 순수 선형 projection 부분을 준비했다.

last-state NPZ를 별도로 제공하면 요청 split마다 `fit_last` 등의 `[N,C]` 값과 `fit_origins` 등의 `[N]` 값, 공통 mean/std가 필요하다. prediction NPZ와 같은 storage scale을 사용한다. mean/std와 origin을 정확히 맞추고 root가 각 last가 해당 과거 창의 마지막 상태에서 왔음을 확인해야 한다. 그 시간 의미까지 이 도구가 원자료에서 재검증하지는 않는다. 기존 FIT teacher의 `last` 하나를 임의로 다른 split에 복사하면 안 된다.

## 실행 인터페이스

이 명령 예시는 승인된 다음 단계에서 root가 경로와 SHA를 고정한 후 실행하는 형식이다. 지금 실제 scientific 자산으로 실행하지 않았다. `--execute-saved-diagnostics` 없이 CLI는 배열을 열지 않는다. 출력 폴더는 반드시 없는 새 경로여야 하며 모든 출력 파일은 exclusive 생성한다.

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:PYTHONDONTWRITEBYTECODE='1'

& .venv/Scripts/python.exe research/smallk_failure_followup_20261014/diagnose_saved_outputs.py `
  --execute-saved-diagnostics `
  --source-manifest 'ROOT_FIXED_SOURCE_MANIFEST.json' `
  --output 'FRESH_OUTPUT_DIRECTORY'
```

manifest의 상대 path는 manifest 자신의 부모 폴더를 기준으로 한다.

```json
{
  "schema": "smallk_saved_diagnostics_v1",
  "dataset": "ROOT_FIXED_DATASET",
  "storage_scale": "normalized",
  "normalization_origin_split": "fit",
  "splits": ["fit", "tune", "val"],
  "split_roles": {
    "fit": "training",
    "tune": "model_selection",
    "val": "reused_development"
  },
  "raw": {"path": "RAW.npz", "sha256": "ROOT_VERIFIED_SHA256"},
  "candidates": [
    {"name": "BASE", "seed": 92601, "path": "BASE_seed1.npz", "sha256": "ROOT_VERIFIED_SHA256"},
    {"name": "BASE", "seed": 92602, "path": "BASE_seed2.npz", "sha256": "ROOT_VERIFIED_SHA256"}
  ],
  "time_bins": 4,
  "tail_quantiles": [0.9, 0.95, 0.99],
  "channels": ["ROOT_FIXED_CHANNEL_NAMES"],
  "k": 4,
  "last": {"path": "ROOT_VERIFIED_PAST_LAST.npz", "sha256": "ROOT_VERIFIED_SHA256"}
}
```

channels는 C개의 고유한 실제 이름이어야 한다. 실제 이름을 확인할 수 없으면 해당 field를 생략하여 channel index를 표시한다. D0만 수행하려면 k와 last를 모두 생략한다. manifest의 모든 source에는 SHA를 필수로 지정한다. 직접 NPZ 형식에서는 `--raw`, 반복 가능한 `--candidate NAME SEED NPZ`, `--dataset`, `--storage-scale`, `--splits`, 반복 가능한 `--split-role SPLIT=ROLE`을 지정한다. 직접 path의 상대 기준은 이 repository root이며 manifest 상대 path의 기준과 다르다.

## 실제 준비 검증

```powershell
& .venv/Scripts/python.exe -m pytest research/smallk_failure_followup_20261014/test_diagnose_saved_outputs.py -q
```

[확인] 최초 CPU threads 2 환경에서 **17 passed in 0.41s, exit 0**이었다. 부분 관측과 전체 결측 채널, zero variance의 저장 scale, 정확 비교 조건, duplicate origin, split/window의 겹침, RAW/own/FIT tail 구분과 gap 재구성, 단위 변환, seed metric 평균과 ensemble 구분, FIT-only basis, zero energy/full rank, source SHA·RAM 제한·TEST/EVAL gate, fresh exclusive 출력을 synthetic 데이터로만 검사했다. 그 뒤 unsupported `test` split이 SHA·ZipFile·np.load 접근 전에 거부되는 검사를 추가했다.

[확인] gate 수정 후 재검증은 **18 passed in 0.32s**였다. CLI `--help`도 정상으로 실행했으며 실제 배열 실행 flag를 주지 않았다.

실제 성능 진단, 그래프·그림 생성, 학습, 새 모델 추론, D1 변환 순서 비교는 이 준비 검증에 포함하지 않았다.
