# 재현과 재개

공개 집계 재생성과 모델을 실제 실행하는 재측정은 필요한 자료가 다르다. 아래 명령은 저장소 루트의 PowerShell에서 실행하며, `<...>`는 실제 위치로 바꾼다. 실행 버전·장치·정밀도는 `ENVIRONMENT.json`, 시작점과 보호 범위는 `START_STATE.json`, 사전 규칙은 `PROTOCOL.md`를 따른다.

## 공개 자료만으로 표·그림 재생성

공개 `receipts/*.receipt.json`에서 원시 반복값을 읽어 CSV·Pareto 표·PNG/SVG를 다시 만들고, 공개 산출물과 바이트 단위로 대사한다. 이어 원시 비용 CSV에서 median·IQR·p95·batch 단위·throughput·allocated/reserved peak·PID·반복 수를 별도로 재계산한다. PyTorch 모델, GPU, 원자료, private 체크포인트, 저장 예측 배열은 이 경로에 필요하지 않다. 같은 NumPy·matplotlib 및 폰트 환경을 사용해야 그림 바이트까지 일치한다.

```powershell
python -m research.memory_budget_20261018_v1.replay `
  --report-root "reports/memory_budget_20261018_v1" `
  --output ".cache/memory_budget_public_replay_<NEW_RUN_ID>"
```

출력은 공개 보고서와 겹치지 않는 **새 경로**로 지정한다. 원 보고서를 출력 위치로 쓰거나 기존 재생성 디렉터리를 재사용하지 않는다. 수치 CSV만 검증할 때에는 `--no-figures`를 추가한다. 성공 시 JSON의 `status=PASS`, 검증 facet 수와 생성 파일별 SHA256을 확인한다. 이 명령은 한국어 해석 문장이나 조건부 학습 결정을 새로 만들지 않는다.

## 실제 실행에 필요한 비공개 자료

Phase A는 새 학습 없이 기존 11개 정확도 실행을 사용한다. Jena는 B2 두 seed·B0 두 seed·Raw·merged LoRA·CPU Ridge, PeMS32는 B2 두 seed·Raw·DirectG다. Raw 두 실행에는 별도 학습 bundle이 없고, 나머지 아홉 개의 bundle/artifact binding과 전체 11개 저장 V 예측을 검증한다. 공유 FP32 Chronos-Bolt-small 스냅샷, Jena·PeMS32 원자료/관측 mask, scaler·채널·origin 계약도 필요하다. 파일 SHA와 모델 tensor-state SHA는 서로 다른 식별자다.

원실험 환경에서는 다음 private cache가 정확한 부모 자료를 연결한다.

```text
.cache/b2_revin_20261018_v1/VERIFIED_ROWS.json
.cache/b2_revin_20261018_v1/LEGACY_INVENTORY.json
.cache/adapter_diagnostics_20261017_v2/
.cache/adapter_improvements_20261018_v1/
<ASSET_ROOT>/.cache/informs_broad_validation_20261015_v1/
.cache/memory_budget_20261018_v1/SOURCE_ROWS.json
.cache/memory_budget_20261018_v1/FACETS_A.json
.cache/memory_budget_20261018_v1/CAMPAIGN.json
```

실제 자산 경로는 기존 binding을 통해 해석하며, 개인 절대경로를 새 공개 파일에 복사하지 않는다. 원자료·private NPZ·체크포인트·거대 캐시는 GitHub에 게시하지 않는다. 따라서 공개 저장소만 clone해서 아래 실제 측정 명령까지 실행할 수 있다는 뜻은 아니다. 부모 공개 집계의 재생성 범위는 [기존 재현 문서](../b2_revin_20261018_v1/REPRODUCE_KO.md)를 참고한다. 이 문서는 누락된 private 자산을 다운로드하거나 학습해 채우지 않는다.

## 원실험의 등록·Phase A 명령

모델 실행은 RTX 4070과 원실험의 고정 Python 환경을 사용한다. 관리 worktree와 원래 사용자 checkout은 구분하며 원래 checkout은 보호 대상으로만 읽는다. 공개된 봉인 파일이나 기존 private cache를 지워 새 실험처럼 시작하면 안 된다. `prepare` 예시는 **해당 캠페인의 최초 등록 명령**이다. 이미 등록한 환경에서는 같은 상태를 확인하며, 다른 컴퓨터의 새 과학 실험은 새 namespace·사전등록이 필요하다. 현재 CLI에는 기존 봉인을 덮어쓰는 옵션이 없다.

```powershell
Set-Location "<EXPERIMENT_WORKTREE>"
$PythonPath = "<ASSET_ROOT>\.venv\Scripts\python.exe"

& $PythonPath -m research.memory_budget_20261018_v1.run prepare `
  --asset-root "<ASSET_ROOT>" `
  --original-root "<ORIGINAL_CHECKOUT>" `
  --attachment "<DOWNLOADS>\CLI_TSFM_Memory_Budget_Experiment_20261018_KO.md" `
  --presentation "<DOWNLOADS>\TSFM_Presentation_Slide_Outline_20261018_KO.md" `
  --presentation "<DOWNLOADS>\TSFM_Presentation_Slide_Outline_20261018_KO (1).md"

& $PythonPath -m research.memory_budget_20261018_v1.run screen --phase A
& $PythonPath -m research.memory_budget_20261018_v1.run confirm --phase A
& $PythonPath -m research.memory_budget_20261018_v1.run gate
```

Phase A screen은 GPU 44개와 CPU Ridge 2개 facet을 실행한다. 각 screen/confirm은 독립 OS 프로세스이며 모델은 한 번 적재한다. 전체 V 일치 검산과 모델 로드는 타이밍 밖이다. screen은 warm-up 2회·측정 3회, 선별 confirm은 10회·20회다. 분할 후보는 clamp 후 중복을 제거한다. 기존 effective-128 경로는 항상 confirm하며, 나머지는 정확도를 보지 않고 측정 메모리·지연으로 선별한다. 후보·제외 이유는 `CONFIRM_SELECTION_A.json`, `RUN_MANIFEST.csv`에서 확인한다.

GPU 실험은 부모의 OS lock과 인증된 child lease 아래 직렬 실행한다. `worker`나 `training` 모듈을 직접 실행해 자원 검증을 우회하지 않는다. 다른 GPU 작업이 있으면 종료하지 말고 `BLOCKED_RESOURCE` 근거를 확인한 뒤 재개한다. PID GPU 메모리 미지원은 `UNAVAILABLE`로 남고 장치 전체 메모리로 대체되지 않는다.

## 조건부 Phase B와 최종 내보내기

`gate`는 `PHASE_B_DECISION.json`을 봉인한다. Jena B2의 확인된 TSFM 경계·반복 메모리 범위 분리·네 신규 모델 전체 및 측정과 마감 여유를 위한 예산이 모두 충족될 때만 아래 Phase B를 진행한다. `train-b`는 봉인 결정이 `PROCEED`일 때 부모 lease를 획득한다. `SKIPPED_GATE`에서는 이유를 출력하고 학습하지 않는다.

```powershell
& $PythonPath -m research.memory_budget_20261018_v1.run train-b

# PROCEED이고 B_TRAINING_SUMMARY.json의 status=COMPLETE일 때만 실행
& $PythonPath -m research.memory_budget_20261018_v1.run screen --phase B
& $PythonPath -m research.memory_budget_20261018_v1.run confirm --phase B

# Phase B를 생략했어도 실행할 최종 공개 집계/그림 내보내기
& $PythonPath -m research.memory_budget_20261018_v1.run export
& $PythonPath -m research.memory_budget_20261018_v1.run preservation
```

Phase B는 Jena K3·K12 각각 두 seed만 별도 학습하고 K6 두 seed는 재사용한다. 공통 referenceK6의 스케줄을 적용해 신규 GPU facet은 20개다. A_fit/A_stop의 네 검색 결정을 모두 봉인한 뒤 fresh A60 refit과 개발 V 평가로 넘어간다. 검색 상한·seed·정밀도·RevIN OFF를 V 결과에 따라 바꾸지 않는다. 하나의 checkpoint에서 K를 즉시 전환하는 실험이 아니다.

`export`는 원실험 private receipt를 공개 가능한 `receipts/`로 옮기고 집계를 재생성하는 명령이므로 private cache가 필요하다. 공개 자료만 있는 경우에는 맨 위의 `replay`를 사용한다. `preservation`도 최초 private 보호 기준선이 있어야 원래 checkout·기존 tracked 파일·발표 자료 보존을 검증할 수 있다.

## 중단 후 재개와 테스트

같은 등록 cache와 생산 소스에서 `screen` 또는 `confirm` 명령을 다시 실행하면 일치하는 COMPLETE 결과를 재사용한다. INVALID 결과는 보존·제외한다. 실제 원인을 검토하고 수정한 뒤 해당 stage의 실패 facet을 다시 시도할 때에는 다음 옵션을 사용한다.

```powershell
& $PythonPath -m research.memory_budget_20261018_v1.run screen --phase A --retry-failed
& $PythonPath -m research.memory_budget_20261018_v1.run confirm --phase A --retry-failed
```

Phase B 측정 재개는 같은 명령의 `--phase B`를 사용한다. 기존 worker가 살아 있으면 PID와 생성 시각으로 재시도를 거부하며, 종료된 실패 receipt·log는 `attempts/`에 보존한다. 학습이 중단되면 같은 `run train-b`를 사용해 보존된 검색/refit checkpoint에서 재개한다. 시간 상한이 지났거나 기존 confirm/gate 봉인과 충돌하면 이 명령이 상한이나 결정을 자동으로 바꾸지 않는다. 근거를 남긴 새 등록이 필요하며, 봉인 파일을 삭제하거나 `SKIPPED_GATE`를 강제로 바꾸는 CLI는 없다. 실제 미완료 여부와 개별 재개 명령은 최종 상태·분석 보고서를 따른다.

```powershell
# GPU 실험이 끝난 뒤 통합 검증
& $PythonPath -m pytest -q tests/test_memory_budget_analysis.py `
  tests/test_memory_budget_replay.py tests/test_memory_budget_run.py `
  tests/test_memory_budget_worker.py tests/test_memory_budget_training.py
```

테스트는 분할 순서·정규화·checkpoint·독립 프로세스·원수치 집계·거부 경로를 검증하며 실제 Phase A의 측정 receipt를 대신하지 않는다. 전체 프로젝트 테스트·실제 공개 replay·원격 게시 검증의 실행 결과와 시간은 최종 validation 산출물을 확인한다. V는 반복 개발 평가이고 기존 T 노출 이력은 유지한다. 그림의 allocated 텐서 메모리와 PID 표본 최대값을 전체 프로세스 hard cap 인증으로 해석하지 않는다.
