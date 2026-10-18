# 재현: 공개 통계·그림, private 예측 재채점, 신규 모델 재실행

이 문서는 실행 방법과 필요한 자산을 설명한다. 아래 명령을 실행했다는 증거나 캠페인의 완료 선언은 아니다. 실제 상태는 해당 실행의 receipt와 검증 JSON을 따른다.

재현 범위를 세 가지로 구분한다.

- **공개 CSV 재현:** GitHub에 공개한 충분통계와 비용 trial만으로 33개 metric, 대응 비교, 42개 비용 집계를 검산하고 PNG/SVG 6개를 실제로 다시 그린다. 원자료·모델 weight·private NPZ가 필요 없다.
- **private 예측 기반 전체 보고서 재현:** 이미 준비된 캠페인의 예측 NPZ와 공개 receipt로 점수·통계·21개 파생 보고서 파일을 다시 만든다. GPU나 모델 재학습은 하지 않는다.
- **모델부터 새로 실행:** canonical 부모 weight와 Jena TRAIN/VAL, 새 시작 등록·보존 baseline·자원 원장이 필요하다. 현재 CLI에는 이를 한 번에 만드는 `init` 명령이나 임의 attempt/output namespace 옵션이 없다. clone 직후 전체 학습이 바로 실행되는 runner로 해석하지 않는다.

보호 TEST의 읽기·해시·요약은 세 경로 모두에 필요하지 않으며 이 문서에서 허용하지 않는다. 반복 개발 VAL·기존 부모 선택 이력·과거 TEST 노출·사전학습 overlap UNKNOWN은 유지된다. 재현 성공을 독립 TEST 확인으로 바꾸어 해석하지 않는다.

## 환경과 의존성 고정

아래 PowerShell 명령은 저장소 루트에서 실행한다. `<...>`는 사용자가 자신의 로컬 값으로 바꾸는 placeholder이며 공개 파일에 실제 개인 경로를 적지 않는다.

```powershell
git clone https://github.com/CanelE452/hierarchical-tsfm-peft.git <FRESH_CHECKOUT>
Set-Location -LiteralPath <FRESH_CHECKOUT>
git switch --detach <PUBLISHED_COMMIT>

py -3.11 -m venv .venv
$py = (Resolve-Path -LiteralPath '.venv/Scripts/python.exe').Path
& $py -c "import sys; assert sys.version_info[:3] == (3, 11, 16), sys.version"
if ($LASTEXITCODE -ne 0) { throw 'Install CPython 3.11.16 before continuing' }
& $py -m pip install -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }

$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
```

`<PUBLISHED_COMMIT>`는 재현할 보고서와 소스가 함께 들어 있는 실제 게시 commit으로 지정한다. `START_STATE.json`의 시작 commit을 최종 게시 commit으로 대신하지 않는다. Linux에서는 CPython 3.11 환경을 만들고 `.venv/bin/python`을 사용한다. 개인 경로나 weight를 Git에 추가하지 않는다.

기준 interpreter는 CPython **3.11.16**, GPU 경로는 FP32·TF32 off·AMP off다. 설치 잠금의 전체 기준은 저장소 루트의 [requirements-lock.txt](../../requirements-lock.txt)이며, 주요 직접 의존성은 다음과 같다.

```text
torch==2.10.0+cu128
chronos-forecasting==2.3.2
transformers==5.17.0
huggingface_hub==1.32.0
numpy==2.4.6
scipy==1.17.1
matplotlib==3.11.2
pillow==12.3.0
fonttools==4.65.0
psutil==7.2.2
pytest==9.1.1
pandas==2.3.3
scikit-learn==1.9.1
safetensors==0.8.0
peft==0.21.0
accelerate==1.15.0
```

잠금 파일은 CUDA 12.8용 PyTorch wheel index도 지정한다. 패키지·폰트·렌더러 버전이 바뀌면 같은 수치를 그려도 image bytes가 달라질 수 있다. 이미지 불일치를 reference 파일 복사로 덮지 않는다. 모델 실행은 설치한 F 구현의 source SHA와 tensor-state SHA도 검증하므로 다른 버전으로 공차를 넓혀 통과시키지 않는다.

## 공개 CSV만으로 재현

clone에 있는 공개 보고서를 source로 사용하고, **존재하지 않는 별도 출력 디렉터리**를 지정한다.

```powershell
$report = 'reports/gt_ts2_fastcheck_20261018_v1'
$publicReplay = Join-Path '.cache/public_report_replays' ([guid]::NewGuid().ToString())

& $py -m research.gt_ts2_fastcheck_20261018_v1.public_replay `
  --source-report $report --output-dir $publicReplay
if ($LASTEXITCODE -ne 0) { throw 'Public replay failed; preserve the actual output and REPLAY_CHECK.json' }
```

입력은 `RUNS.csv`, `COMPARISONS.csv`, `CHANNEL_HORIZON_STATS.csv`, `COST.csv`, `COST_TRIALS.csv`와 공개 reference image 6개다. `cache://...` 문자열이 CSV에 남아 있어도 private 파일을 열지 않는다.

- 채널별 TOTAL SSE/SAE/count를 FP64로 재집계하여 RUNS의 channel-macro MSE/MAE와 대사한다.
- 1–16/17–32/33–48의 충분통계 합이 TOTAL을 복원하는지 검사한다.
- 공개 cohort fingerprint의 동일성과 반올림 전 상대 변화 `(candidate-reference)/reference*100`을 대사한다.
- 비용마다 warmup 10회·timed 20회·고정 origin 그룹 3개 순환을 검사하고 p50/IQR/p95·allocated/reserved/RSS 집계를 다시 계산한다. 이 quantile은 서비스 tail 지연 보장이 아니다.
- 같은 plotting 코드로 `FIG1_Lookback`, `FIG2_Adaptation`, `FIG3_Cost`의 PNG/SVG를 새로 만든 뒤 공개 image와 SHA256을 비교한다.

출력은 실제 생성 image 6개와 `REPLAY_CHECK.json`이다. JSON은 입력 CSV SHA, 독립 집계 범위, 기대/실제 image SHA를 담는다. PASS는 exit 0, FAIL은 exit 1이다. mismatch가 나도 실제 생성 image와 FAIL 증거를 보존하며 reference image를 복사해 통과시키지 않는다. 공개 통계의 재집계는 private 원자료가 올바르게 생산되었다는 별도 증명을 대체하지 않는다.

## 이미 준비된 private 예측으로 전체 보고서 재현

이 경로는 완료·부분완료 상태를 보존한 **기존 캠페인 evidence**를 읽는 CPU 작업이다. source report와 source cache를 명시하고 새 출력 디렉터리를 쓴다.

필요한 private 입력:

```text
<PRIVATE_CAMPAIGN_CACHE>/GPU_LEDGER.json
<PRIVATE_CAMPAIGN_CACHE>/predictions/<run_id>.npz
```

각 NPZ에는 `prediction,target,mask,origins,mean,std`가 있어야 한다. 공개 쪽에는 해당 `receipts/`, `training/`, `cost_receipts/`, `stage_receipts/`와 START/contract/parent/update/resource 증거가 필요하다. 이 재채점 경로는 모델 weight나 원 Jena archive를 다시 읽을 필요가 없다.

```powershell
$report = 'reports/gt_ts2_fastcheck_20261018_v1'
$privateCache = '<PRIVATE_CAMPAIGN_CACHE>'
$fullReplay = Join-Path '.cache/full_report_replays' ([guid]::NewGuid().ToString())

& $py -m research.gt_ts2_fastcheck_20261018_v1.reporting `
  --source-report $report --source-cache $privateCache --output-dir $fullReplay
if ($LASTEXITCODE -ne 0) { throw 'Private prediction replay failed' }
```

원 보고서의 image·CSV·Markdown·JSON과 파생 파일 21개의 bytes를 비교한다. `REPRODUCE.md`, publication proof, 원 receipt는 파생 21개에 포함되지 않는다.

```powershell
@'
import sys
from pathlib import Path
from research.gt_ts2_fastcheck_20261018_v1 import reporting, common
source, cache, output = map(Path, sys.argv[1:])
result = reporting.generate(output, source_report=source, source_cache=cache)
mismatches = [name for name, sha in result["files"].items()
              if not (source / name).is_file() or common.file_sha(source / name) != sha]
print(common.json_bytes({"derived_files": len(result["files"]),
                         "mismatches": mismatches, "source_status": result["status"]}).decode())
raise SystemExit(1 if mismatches else 0)
'@ | & $py - $report $privateCache $fullReplay
if ($LASTEXITCODE -ne 0) { throw 'Derived report bytes differ; preserve both attempts' }
```

위 비교 스크립트는 앞서 만든 동일 bytes의 replay 출력을 다시 읽는다. 다른 내용의 기존 파일은 덮어쓰지 않는다. `FINAL_RESOURCE_SNAPSHOT.json`과 기존 stage 종료시각을 사용하므로 replay 시간을 과거 GPU 원장에 더하지 않는다.

## CPU 회귀 테스트 범위

Jena 실제 배열 fingerprint 검사까지 포함하려면 TRAIN/VAL archive가 있는 private asset root를 명시한다. 지정하지 않으면 실제 자산용 테스트가 skip되므로, skip을 실제 배열 검증 완료로 보고하지 않는다.

```powershell
$env:GT_TS2_ASSET_ROOT = '<PRIVATE_ASSET_ROOT>'
$env:CUDA_VISIBLE_DEVICES = '-1'
New-Item -ItemType Directory -Force -Path '.cache' | Out-Null
$testLog = Join-Path '.cache' ('gt_ts2_cpu_regression_' + [guid]::NewGuid().ToString() + '.log')

& $py -m pytest -q `
  --ignore=tests/test_model_parity.py `
  --ignore=tests/test_baselines.py *> $testLog
$testExit = $LASTEXITCODE
if ($testExit -ne 0) { throw 'CPU regression failed; retain the full log and actual failure names' }
```

제외 대상은 다음 2개 파일이다.

- `tests/test_model_parity.py`: `hier_peft.smoke.run_smoke()`를 호출하는 기존 Labour GPU integration이며, 기존 smoke checkpoint·UPDATE_LEDGER·SMOKE_AUDIT를 쓴다. 이번 CPU 회귀에서 보호 대상인 기존 산출물을 갱신하지 않기 위해 제외한다.
- `tests/test_baselines.py`: Labour 실제 데이터가 필요한 다른 작업의 통계 baseline integration이다. 이번 Jena GT/TS2 검증 범위와 다르다.

이는 repository 전체 테스트가 조건 없이 성공한다는 주장이 아니다. 실제 exit code, pass/skip/fail 수, 전체 로그 SHA, 제외 파일과 이유를 `TEST_VALIDATION.json`에 기록한다. GPU cost의 timed 구간과 병행 실행하지 않는다.

## 모델부터 새로 실행할 때 필요한 private 자산

`asset_root`는 canonical 부모 산출물을 둔 읽기 전용 저장원이며, 새 campaign 출력과 분리한다. 필요한 Jena archive는 다음 1개다.

```text
.cache/tsfm_peft_practical_controls_v7_20260928/data/jena_trainval.npz
SHA256 29cfae52d7cd7128e551ef85da11138ed4d3f7b3ccf27af359c4fc1dedf4a933
```

필요한 canonical run ID는 다음 4개다.

```text
jena_GT_full_s92601
jena_GT_full_s92602
jena_TS2_d320_s92601_retry_batch4_v4
jena_TS2_d320_s92602_retry_batch4_v4
```

각 run의 private base는 `.cache/professor_compressor_direct_20261011/runs/<run_id>/`이다. GT는 `forecast_calibrated_best.pt`, TS2는 `encoder_swa.pt`와 `forecast_calibrated_decoder_best.pt`, 각각의 `prediction.npz`·`diagnostic.npz`가 필요하다. 공개된 `research/professor_compressor_direct_20261011/`의 `protocol.json`, `runs.json`, 각 `result.json`, historical source snapshot, vendor code도 SHA로 검증한다. GT full을 no_spatial로, TS2 d320을 d16이나 다른 retry로 대체하지 않는다.

F는 다음과 동일한 Chronos-Bolt-small snapshot을 local-files-only로 읽는다.

```text
.cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8
tensor-state SHA256 354fe0801e121851c0a9886349cab5939f1059a292f019f6a197924d485e7965
```

선택 checkpoint SHA·decoder SHA·source pin은 [PARENT_BINDINGS.json](PARENT_BINDINGS.json)과 원 `result.json`을 기준으로 한다. `PRIVATE_PATHS.json`의 `model_directory`로 local snapshot을 지정할 수 있지만, weight·구현을 다른 model/version으로 바꾸는 설정은 아니다.

## 신규 모델 attempt 시작 등록과 stage 순서

새 model fit은 위 두 replay와 구분하여 독립적으로 등록한다. 새 worktree라도 checkout한 공개 report가 canonical output path에 있으면 `seal_json`이 충돌한다.

```powershell
git worktree add --detach <FRESH_MODEL_WORKTREE> <PUBLISHED_COMMIT>
Set-Location -LiteralPath <FRESH_MODEL_WORKTREE>
git status --short
```

새 worktree에는 기존 `.venv`가 자동 복사되지 않는다. 이 worktree에서도 위 환경 준비 절을 먼저 적용한다.

다음은 **시작 전 최소 준비 recipe**이며, 구현되지 않은 init CLI를 뜻하지 않는다.

1. 새 worktree 안에서만 공개 report를 별도 reference 위치에 bytes를 보존하여 옮기고, 원 checkout과 공개 evidence는 변경하지 않는다. canonical `reports/gt_ts2_fastcheck_20261018_v1/`과 `.cache/gt_ts2_fastcheck_20261018_v1/`이 새 attempt 전용인지 확인한다. 기존 report나 실패 cache 위에서 시작하지 않는다.
2. 같은 code package의 `common.CODE/REPORT/CACHE`는 고정 namespace에서 결정된다. `run`에 임의 `--attempt`, `--asset-root`, `--model-directory`를 전달하는 기능은 없다. namespace/output 분리를 먼저 준비하며, package 이름을 기계적으로 바꾸어 source pin을 우회하지 않는다.
3. private `PRIVATE_PATHS.json`에 `asset_root`, `python`, `model_directory`를 등록한다. 실제 로컬 값은 private cache에만 둔다. 환경변수 `GT_TS2_ASSET_ROOT`는 CPU asset test용이며 runner의 이 설정을 대체하지 않는다.
4. private `GPU_LEDGER.json`은 새 시작 시에만 `{"status":"IDLE","GPU_seconds":0.0,"stages":[]}`로 만든다. 실패·재개 시 원장을 0으로 되돌리지 않는다.
5. 실제 시작시각·시작/remote SHA·승인 spec SHA·예산을 새 `START_STATE.json`에 기록한다. `PROTECTED_FILES.json`에는 보호할 기존 파일의 bytes/SHA를 저장하되 TEST archive는 **열기 전에 제외**한다. `ORIGINAL_PRESERVATION_BASELINE.json`은 원 checkout HEAD/status SHA와 보호할 presentation baseline을 가진다. 원 시작 receipt를 새 실행의 시작 정보로 복사하지 않는다.
6. source history/amendment, 검증할 예산·updates·최종 proof schema를 새 attempt에 맞게 확인한다. 기존 `verify.py`는 공개한 특정 캠페인의 300 updates·source lineage·시작시각/90분GPU/3시간wall·최종 proof에 특화돼 있다. 100-update pilot이나 다른 시작 등록을 자동 인증하는 portable generic verifier가 아니다.

private `PRIVATE_PATHS.json`의 최소 형태는 다음과 같다. placeholder를 실제 로컬 값으로 채운 이 파일을 공개 report에 복사하지 않는다.

```json
{
  "asset_root": "<PRIVATE_ASSET_ROOT>",
  "python": "<PRIVATE_PYTHON_EXECUTABLE>",
  "model_directory": "<PINNED_BOLT_SNAPSHOT_DIRECTORY>"
}
```

등록을 마친 새 attempt에서 다음 순서로 stage를 실행한다. GPU 작업은 동시에 1개이며 다른 campaign·사용자 process를 중단하지 않는다. 새로운 F/TS2 사전학습이나 GT reconstruction 재학습은 포함하지 않는다.

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUDA_VISIBLE_DEVICES = '0'
$py = (Resolve-Path -LiteralPath '.venv/Scripts/python.exe').Path

foreach ($stage in @('prepare', 'validate', 'features', 'smoke')) {
    & $py -m research.gt_ts2_fastcheck_20261018_v1.run $stage
    if ($LASTEXITCODE -ne 0) { throw ('Stopped at stage: ' + $stage) }
}
```

`prepare`는 TRAIN/VAL·부모 binding·계약을 봉인한다. `validate`는 A0 historical L512 재현, shape, forward/input-gradient를 확인한다. `features`는 PRE/RAND별 TRAIN feature 통계와 고정 TS2 feature를 만든다. `smoke`는 seed92601의 4 arm×20 TRAIN updates를 버리고, VAL을 보기 전에 전체 8 arm 공통의 300/100/NOT_RUN 예산 결정을 봉인한다.

`UPDATE_DECISION.json`을 읽고 전체 8 arm과 마지막 20%를 유지하는 등록된 판단인지 확인한 뒤 진행한다. 성능이 나쁘다는 이유로 updates를 바꾸지 않는다. `NOT_RUN_BUDGET`을 실행 성공과 구분해 보고한다.

```powershell
& $py -m research.gt_ts2_fastcheck_20261018_v1.run phase_a
if ($LASTEXITCODE -ne 0) { throw 'Phase A stopped' }
& $py -m research.gt_ts2_fastcheck_20261018_v1.run phase_b
if ($LASTEXITCODE -ne 0) { throw 'Phase B stopped' }
```

Phase B는 부모/worker 양쪽에서 현재 예산을 재검사하며, 같은 seed의 4 arm에 동일한 197-origin sampling order와 마지막 5개 batch를 사용한다. CP0와 고정 최종 update를 남기고 VAL로 checkpoint를 다시 고르지 않는다. 300 updates 종료 자체가 수렴 증명은 아니다.

Phase B와 필요한 예측 출력이 존재하는지 확인하고 cost를 실행한다. `cost`는 Phase A 13 route + 최종 Phase B 8 route 각각의 batch1/8을 **별도 OS process**에서 측정하는 coordinator다. 개별 `cost --run-id ... --batch ...`를 병렬 실행하지 않고 이 coordinator를 사용한다.

```powershell
& $py -m research.gt_ts2_fastcheck_20261018_v1.run cost
if ($LASTEXITCODE -ne 0) { throw 'Cost stopped; retain its independent worker log' }

& $py -m research.gt_ts2_fastcheck_20261018_v1.observation
if ($LASTEXITCODE -ne 0) { throw 'Observation diagnostic stopped' }
```

observation은 양쪽 history의 모든 cell이 관측된 mask-only subset의 보조 기술통계이며, 주평가 371개 원점을 교체하지 않는다. 공개한 시도의 엄격한 subset rule은 Phase A 이후의 사후 진단으로 기록한다. 새 실행에서도 등록시점을 정직하게 남긴다.

GPU lease가 IDLE이 된 뒤 실제 `FINAL_RESOURCE_SNAPSHOT.json`을 고정하고, CPU 회귀와 실제 로그 proof, 공개/전체 private replay proof를 기록한다. 기존 proof를 새 시도에 재사용하지 않는다.

```powershell
& $py -m research.gt_ts2_fastcheck_20261018_v1.run report
if ($LASTEXITCODE -ne 0) { throw 'Report generation stopped' }

& $py -m research.gt_ts2_fastcheck_20261018_v1.verify --check-only
```

최종 verifier는 `python -m research.gt_ts2_fastcheck_20261018_v1.verify`를 직접 호출한다. 기존 `run verify` wrapper는 추가 인자를 전달하는 범용 CLI가 아니다. `--check-only`는 누락 evidence의 목록이며 인증이 아니다. 필요한 evidence와 새 시도에 맞는 검증 계약이 갖춰질 때까지, 생성된 보고서가 있다는 이유만으로 validation PASS를 선언하지 않는다. 과거 캠페인의 elapsed budget을 현재시각으로 검사하는 finalizer를 공개 clone의 영구 replay 명령으로 사용하지 않는다.

## 실패·재개와 보존 규칙

receipt, NPZ, checkpoint, stage log는 원칙적으로 `xb`/seal로 불변을 유지한다. report/public replay도 별도의 새 출력에 만든다. 같은 bytes의 기존 파일은 재사용할 수 있지만 다른 기존 산출물을 조용히 갱신하지 않는다.

- 실패한 stage log·부분 checkpoint·CP0를 보존한다. exit code와 중단 이유를 남긴다.
- `GPU.lock`이나 RUNNING 원장만 보고 자동 삭제하지 않는다. 실제 process와 원장을 확인하고 다른 사용자의 job을 종료하지 않는다.
- stage마다 재개 동작이 다르다. 기존 COMPLETE receipt를 skip하는 처리는 있지만, 중간의 sealed CP0·checkpoint·실패 log를 덮어 재시도하는 범용 resume는 아니다. 새 attempt와 등록을 준비한 뒤 재개한다.
- OOM 대응은 등록 effective batch를 유지하는 microbatch 조정 범위로 제한한다. 길이·정밀도·origin 수·seed를 조용히 줄이지 않는다.
- 부모 A0 재현·데이터/scale/cohort·input gradient·frozen state 불일치를 수치상 승패로 처리하지 않는다. 검증 실패, 미실행, 정상 실행했지만 미개선인 경우를 구분한다.
- 원 산출물, 원 presentation, 기존 budget ledger, 보호 TEST를 변경하지 않는다. 모델 rerun·replay를 실행했다는 사실에서 새 TEST 평가 허가를 추론하지 않는다.
