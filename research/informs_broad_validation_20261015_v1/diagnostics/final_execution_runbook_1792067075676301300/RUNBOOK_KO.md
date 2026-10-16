# GPU 반환 후 final 실행 runbook

기준 시각: 2026-10-15T12:24:35.676301+00:00. [확인] 현재 소스의 API/조건을 읽어 작성했다. 이 문서 작성 중 GPU 실행·자료/모델 로드·검사 실행·소스 변경·Git 변경은 없었다. parser 도움말만 AST에서 argparse 구문을 분리해 렌더했으며 campaign/torch/numpy 모듈을 import하지 않았다. 전체 학습·최종 결과·게시가 완료됐다는 문서가 아니다.

실제 최소 순서는 **비용 → V checkpoint replay → 선택 전 독립 CPU 감사 → 명시적 selection lock → locked T → E0 B 실제 재추론 → 선택 후 독립 CPU 감사 → 보고서**다. 비용과 replay가 모두 선택의 근거가 되며 T 결과로 방법을 다시 고르지 않는다. 선택 전 감사가 없으면 `lock_selection`이 거부한다.

입력 소스 26개의 SHA는 [INPUT_SOURCES.sha256](INPUT_SOURCES.sha256)에 있다. source6 고정과 별개로 `final_evaluation.SOURCE_FILES` 18개 전체가 selection lock에 들어간다. 잠금 후 이 18개 중 하나만 바뀌어도 T gate가 닫힌다. 이 runbook은 resources/cost/legacy/report까지 추가로 pin한다.

## 0. GPU 반환·source freeze·입력 결정

ROOT가 기존 학습 owner/모든 child의 종료를 확인하고 난 뒤 시작한다. `ACTIVE_GPU_LEASE.json` 원문에는 인증 token이 있으므로 출력하거나 공개 문서에 복사하지 않는다. live owner가 남아 있으면 새 reservation을 시도하지 않는다. registry에 RUNNING이 남아 있으면 원 실행의 종료 근거에 따라 ROOT가 상태를 정리해야 하며, 완료로 일괄 변경하지 않는다.

저장소 루트의 PowerShell에서 다음 환경을 실제 기존 경로로 먼저 설정한다: `INFORMS_ASSET_ROOT`, `INFORMS_CACHE_ROOT`, `INFORMS_PYTHON`. 값은 이 문서에 적지 않는다. GPU 단계에는 ROOT가 소유할 GPU의 `CUDA_VISIBLE_DEVICES`를 설정하며 CPU 검사 때의 `-1`을 그대로 넘기지 않는다.

```powershell
$repo = (git rev-parse --show-toplevel).Trim()
Set-Location (Join-Path $repo 'research/informs_broad_validation_20261015_v1')
$r = (Get-Location).Path
$py = $env:INFORMS_PYTHON
if (-not $py -or -not $env:INFORMS_ASSET_ROOT -or -not $env:INFORMS_CACHE_ROOT) { throw '기존 실행 환경을 먼저 설정해야 함' }
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'; $env:OPENBLAS_NUM_THREADS='2'
$runbook = Join-Path $r 'diagnostics/final_execution_runbook_1792067075676301300'
Get-Content -LiteralPath (Join-Path $runbook 'INPUT_SOURCES.sha256') | ForEach-Object {
    $pair = $_ -split '  ',2
    $actual = (Get-FileHash -LiteralPath (Join-Path $r $pair[1]) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $pair[0]) { throw "runbook 기준 source 변경: $($pair[1])" }
}
```

필요 입력은 ROOT가 명시적으로 고정한 private JSON 하나다. 그 경로를 `INFORMS_FINAL_EXECUTION_PLAN`에 설정한다. 아래는 필드 형식이며 빈 목록을 자동 선택하거나 모든 registry 행으로 확장하지 않는다.

```json
{
  "cost_candidates": {"dataset_name": ["registered_run_id"]},
  "replay_run_ids": ["registered_run_id"],
  "selected_run_ids": ["registered_run_id"],
  "legacy_ids": ["actual_legacy_id"],
  "decision": {
    "stop_training": true,
    "reason": "실제 종료 근거, V/비용 선택 규칙, 미완료 범위를 명시",
    "audit_report_relative_path": "audits/audit_actual_id/AUDIT_REPORT.json"
  }
}
```

선택 목록은 V·비용·replay만으로 결정한다. `selected_run_ids`는 replay PASS인 후보 중 ROOT가 고른 명시 목록이어야 한다. baseline/대조군의 T 비교가 필요하면 잠금 전에 같은 목록에 넣어야 한다. T 뒤에 비교 대상을 추가할 수 없다. 후보 ID·dataset은 실제 registry/V manifest와 대조한다.

후보 V 제한:

- 비용의 accuracy binding은 실제 `predictions/<id>/V_select/`, `ordered_predictions/<id>/`, `route_predictions/<id>/V_select/` 중 manifest가 있어야 한다. 실제 선택 artifact·정규화·모델·source SHA가 결박된다. 미실행/계획 행만 비용 후보로 넘기지 않는다.
- 현재 `route_predictions._standard`의 replay 입력은 **predictions 또는 ordered_predictions 두 위치뿐**이다. route_predictions V만 있는 ID는 비용/lock에서는 발견되더라도 현재 replay API 입력으로는 지원되지 않는다. 임의 복사로 원본 V를 꾸미지 말고 그 후보를 중단 사유로 남긴다.
- `lock_selection`은 registry status가 COMPLETE_VALID 또는 PARTIAL로 시작하는 실제 실행 행만 받는다. audit가 수용하는 BUDGET_LIMIT/LIMITED_SCREEN과 동일한 집합이 아니다.
- `_V_manifest`는 predictions → ordered → route_predictions 순으로 첫 파일을 택한다. 같은 ID의 다른 V 사본을 마음대로 선택하는 API가 아니다.

## 1. 비용: cost_campaign CLI가 owner를 직접 연다

현재 parser의 실제 인자는 다음뿐이다. `--cache-root`, `--public-root`, `--epochs`, `--budget`, `--run-ids`는 이 CLI 인자가 아니다.

```text
usage: cost_campaign.py [-h] --dataset DATASET --run-id RUN_ID
                        [--skip-raw-variants]
                        [--reserve-final-seconds RESERVE_FINAL_SECONDS]
```

ROOT가 직접 도움말을 볼 명령은 `& $py -X utf8 -B cost_campaign.py --help`다. 이 문서 작성에서는 실제 campaign import를 피하고 같은 argparse AST만 렌더했다. `--dataset`은 한 dataset, `--run-id`는 ID마다 반복한다. final 전 기본 reserve는 3600초이며 유효 범위는 유한한 [0,43200)이다.

```powershell
$plan = Get-Content -LiteralPath $env:INFORMS_FINAL_EXECUTION_PLAN -Raw | ConvertFrom-Json
foreach ($entry in $plan.cost_candidates.PSObject.Properties) {
    $costArgs = @('-X','utf8','-u','-B','cost_campaign.py','--dataset',$entry.Name,'--reserve-final-seconds','3600')
    foreach ($rid in $entry.Value) { $costArgs += @('--run-id',[string]$rid) }
    & $py @costArgs
    if ($LASTEXITCODE -ne 0) { throw '비용 실행 오류: 보존 로그/receipt 확인' }
}
```

**이 명령을 외부 GPUReservation 안의 child로 실행하면 안 된다.** CLI가 자체 `GPUReservation(CACHE,total_gpu_seconds=43200-reserve,purpose='audit',reserve_seconds=0,...)`를 연다. 이미 owner가 있는 ROOT가 API로 비용을 실행하려면 CLI를 다시 시작하지 않고 `run_conditions(routes,private,public,lease,...)`에 동일 owner를 넘겨야 한다. 이때 후보 lock·고유 출력 디렉터리·조건계획의 준비도 ROOT 책임이다.

[확인] 비용 기본 조건은 B1/B8 각각 fresh process와 device/model-size별 balanced 세션, warmup10/timed20/stage20이다. Raw는 full/32/8/4 fresh row 변형을 보존하고 같은 batch의 성공 변형 중 가장 빠른 것을 balanced에 넣는다. `--skip-raw-variants`는 이 범위를 줄이므로 full protocol을 채웠다는 문구와 함께 쓰지 않는다. CPU ridge/current/seasonal은 별도 CPU scope, direct G 계열은 CUDA/F0일 수 있다. shared-F balanced memory와 fresh worker memory를 섞지 않는다.

[확인] 43200은 금액이나 명령마다 새로 받는 시간 예산이 아니다. shared `GPU_USAGE.jsonl`의 누적 exclusive owner process-wall 초다. audit 목적에서는 `reserve_seconds`를 차감하지 않으므로 cost CLI는 **total ceiling 자체를 39600으로 낮춰** 3600초를 남긴다. 기존 charge, probe, 사고의 보수 charge를 제거하지 않는다. owner 안 CPU child/대기/close 시간도 charge에 포함된다. child 남은 시간 <30초, CostLease <15초에서는 추가 작업을 보류하며 이미 제출된 CUDA 연산을 강제 취소하는 장치는 아니다. wall 상한도 `1792050350+172800`으로 유지된다.

각 새 `costs/<dataset>_<time>/COST_CAMPAIGN_RECEIPT.json`을 확인한다. exit0만으로 완료를 판정하지 않는다. 필요한 조건은 `actual_full_protocol_completed=true`, 각 child의 `COMPLETE_PROTOCOL`, `BOUND_EXACT_ROUTE`, SHA가 결박된 raw trials다. DEFERRED_RESOURCE/PARTIAL_CONDITION_FAILURES는 실패·미완료로 보존한다. reserve0은 T 뒤 남은 비용을 채울 때만 명시적으로 선택하며 누적43200을 늘리지 않는다.

## 2. V checkpoint replay: 외부 audit owner를 API에 넘긴다

`final_evaluation.py`에는 main/CLI가 없다. `python final_evaluation.py`는 실행 요청이 아니다. 실제 API는 `execute_replays(run_ids,cache,public,lease,deadline)`이다. 아래 owner는 다른 owner가 완전히 닫힌 뒤 연다. 이 owner를 또 child의 GPUReservation으로 감싸지 않는다.

```powershell
@'
from pathlib import Path
import os,json
import final_evaluation as f
from resources import GPUReservation
p=Path.cwd(); c=Path(os.environ["INFORMS_CACHE_ROOT"])
plan=json.loads(Path(os.environ["INFORMS_FINAL_EXECUTION_PLAN"]).read_text(encoding="utf-8-sig"))
ids=plan["replay_run_ids"]
assert ids and len(ids)==len(set(ids))
with GPUReservation(c,total_gpu_seconds=43200,reserve_seconds=0,
                    wall_deadline=1792050350+172800,purpose="audit",kind="final_V_checkpoint_replay") as lease:
    result=f.execute_replays(ids,c,p,lease,lease.deadline-5)
print(json.dumps(result,ensure_ascii=False))
assert result["status"]=="COMPLETE_SELECTED_ROUTES" and result["failed"]==0 and not result["deferred_run_ids"]
assert all(row["status"]=="PASS" for row in result["records"])
'@ | & $py -X utf8 -B -
if ($LASTEXITCODE -ne 0) { throw 'V replay 결과를 확인하고 선택을 중단' }
```

원점은 V first/middle/last+seed92601의 추가 최대3개이며 subset이다. 기본 tolerance는 atol1e-5/rtol1e-4다. CPU 산술 감사나 학습 재현·full-V 새 forward로 세지 않는다. receipt가 있으면 exact pre-inference/source/route/cache binding이 같을 때 재사용하므로 `executed_now`를 따로 집계한다. FAIL_NUMERICAL_PARITY를 성공으로 덮거나 허용오차를 즉석에서 바꾸지 않는다. wrapper가 독립 route 실패 뒤 다음 route를 계속할 수 있으므로 최종 batch records를 모두 확인한다.

## 3. 선택 전 CPU 감사 → lock: GPU owner를 닫고 실행

이 단계에는 `CUDA_VISIBLE_DEVICES=-1`, OMP/MKL/OPENBLAS=2를 child에 명시한다. 감사는 저장 예측 NPZ를 읽지만 새 학습/F/timer를 실행하지 않는다. 이 문서 작성에서는 실행하지 않았다.

```powershell
$gpuVisibility = $env:CUDA_VISIBLE_DEVICES
$env:CUDA_VISIBLE_DEVICES='-1'
@'
from pathlib import Path
import os,json
import audit_campaign
p=Path.cwd(); c=Path(os.environ["INFORMS_CACHE_ROOT"])
report=audit_campaign.audit_all(p,c)
print(json.dumps(report,ensure_ascii=False))
assert report["errors"]==[]
assert report["status"].startswith("PASS_AVAILABLE_PREDICTIONS")
print("AUDIT_REPORT_RELATIVE_PATH=audits/"+report["audit_batch"]+"/AUDIT_REPORT.json")
'@ | & $py -X utf8 -B -
if ($LASTEXITCODE -ne 0) { throw '선택 전 독립 감사 불통과' }
```

이 실제 audit 경로를 private plan의 `decision.audit_report_relative_path`에 고정한다. ROOT가 V·비용·PASS replay 근거로 선택한 IDs와 종료 이유를 확인한 뒤 lock API를 부른다. lock은 자동 방법 선택을 하지 않는다. 잠금 뒤 현재 캠페인의 새 학습·레지스트리 과학 조건 추가는 재개하지 않는다.

```powershell
@'
from pathlib import Path
import os,json
import final_evaluation as f
p=Path.cwd(); c=Path(os.environ["INFORMS_CACHE_ROOT"])
plan=json.loads(Path(os.environ["INFORMS_FINAL_EXECUTION_PLAN"]).read_text(encoding="utf-8-sig"))
ids=plan["selected_run_ids"]; assert ids
assert set(ids)<=set(plan["replay_run_ids"])
for rid in ids:
    rec=json.loads((p/"route_replays"/rid/"REPLAY_RECEIPT.json").read_text())
    assert rec["status"]=="PASS"
decision=plan["decision"]
assert decision["stop_training"] is True and decision["reason"].strip()
assert decision["audit_report_relative_path"]
lock=f.lock_selection(ids,decision,p,c)
print(json.dumps({"status":lock["status"],"selection_sha256":lock["selection_sha256"],
                  "allowed_run_ids":lock["allowed_run_ids"],
                  "mandatory_coverage_complete":lock["mandatory_coverage_complete"]},ensure_ascii=False))
assert lock["locked"] is True and lock["source_bindings"]==f._sources()
'@ | & $py -X utf8 -B -
if ($LASTEXITCODE -ne 0) { throw 'selection lock 불통과' }
$env:CUDA_VISIBLE_DEVICES=$gpuVisibility
```

[확인] lock_selection 자체는 replay receipt PASS를 강제하지 않는다. 그래서 위 ROOT gate가 필요하다. lock은 선택 V를 다시 산술 검산하고 audit source SHA·선택 manifest exact SHA·독립 records를 확인한다. registry RUNNING/live training lease를 거부한다. PARTIAL 캠페인의 잠금도 가능하지만 `LOCKED_PARTIAL_CAMPAIGN`과 mandatory_incomplete를 유지한다. 기존 SELECTION_LOCK이 있으면 선택/decision 교체가 금지되며 정확히 같은 기존 lock을 검증해 재사용해야 한다. 단순히 새 audit 경로로 decision을 바꾸어 재잠그지 않는다.

## 4. locked T: source SHA 재확인 후 실행

source manifest 검사를 다시 수행한다. 잠금의 18 source·registry scientific identity·선택 V/reference/archive·artifact/public binding을 바꾸지 않는다. GPU visibility를 ROOT의 실제 장치로 복구했는지 확인한다.

```powershell
@'
from pathlib import Path
import os,json
import final_evaluation as f
from resources import GPUReservation
p=Path.cwd(); c=Path(os.environ["INFORMS_CACHE_ROOT"])
assert (p/"SELECTION_LOCK.json").is_file()
with GPUReservation(c,total_gpu_seconds=43200,reserve_seconds=0,
                    wall_deadline=1792050350+172800,purpose="audit",kind="locked_T_final") as lease:
    f._validated_lock(c,p)
    result=f.execute_T(c,p,lease,lease.deadline-5)
print(json.dumps(result,ensure_ascii=False))
assert result["status"]=="COMPLETE_SELECTED_ROUTES" and result["failed"]==0 and not result["deferred_run_ids"]
'@ | & $py -X utf8 -B -
if ($LASTEXITCODE -ne 0) { throw 'T 실행 미완료: batch/보존 오류 확인' }
```

`execute_T`는 lock의 allowed_run_ids만 사용한다. 호출자가 별도 T 후보 목록을 넘기지 않는다. full registered T cohort를 실제 물리 입력으로 실행하고 `route_predictions/<id>/T_final/`에 저장하며 동일 artifact는 재사용할 수 있다. T 전 locked source gate가 task/T 접근보다 먼저 실행된다. gate를 우회해 `route_predictions.predict_route`를 직접 실행하지 않는다. 결과가 PARTIAL_ROUTE_FAILURES/DEFERRED_RESOURCE이면 해당 batch와 pending 범위를 보고한다.

## 5. E0 B: 별도 fresh child에 owner 환경을 전달한다

현재 API는 `legacy_evaluation.evaluate_legacy(asset_root,cache_root,...)`, CLI는 아래 인자다. `inventory`도 실제 저장 checkpoint를 CPU에서 읽으므로 순수 metadata 명령으로 오해하지 않는다. 이 runbook 작성에서는 inventory를 실행하지 않았다.

```text
legacy_evaluation.py {inventory,run} --asset-root ASSET_ROOT --cache-root CACHE_ROOT
                     [--id IDS] [--batch-size BATCH_SIZE] [--row-microbatch ROW_MICROBATCH]
                     [--execute-evaluation] [--result-json RESULT_JSON]
```

`run`은 `--execute-evaluation`과 authenticated central child lease를 요구한다. default batch8이며 row-microbatch는 양수다. 다음 child는 새로운 owner/별도 예산을 만들지 않는다. 모델 로딩을 다른 단계와 격리하고 child가 끝난 뒤 owner를 닫는다.

```powershell
@'
from pathlib import Path
import os,json,time,sys,subprocess
from resources import GPUReservation
p=Path.cwd(); c=Path(os.environ["INFORMS_CACHE_ROOT"]); a=Path(os.environ["INFORMS_ASSET_ROOT"])
plan=json.loads(Path(os.environ["INFORMS_FINAL_EXECUTION_PLAN"]).read_text(encoding="utf-8-sig"))
assert plan["legacy_ids"]
stamp=str(time.time_ns()); folder=c/"final_execution"/stamp; folder.mkdir(parents=True,exist_ok=False)
output=folder/"E0_B_RESULT.json"; log=folder/"E0_B.log"
cmd=[sys.executable,"-X","utf8","-u","-B",str(p/"legacy_evaluation.py"),"run",
     "--asset-root",str(a),"--cache-root",str(c),"--batch-size","8",
     "--row-microbatch","128","--execute-evaluation","--result-json",str(output)]
for rid in plan["legacy_ids"]: cmd.extend(["--id",rid])
with GPUReservation(c,total_gpu_seconds=43200,reserve_seconds=0,
                    wall_deadline=1792050350+172800,purpose="audit",kind="legacy_E0_B_reinference") as lease:
    env=dict(os.environ,**lease.child_environment(),OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
    with log.open("xb") as stream:
        code=subprocess.Popen(cmd,cwd=p,env=env,stdout=stream,stderr=subprocess.STDOUT).wait()
assert code==0 and output.is_file()
result=json.loads(output.read_text(encoding="utf-8-sig"))
print(json.dumps({k:result[k] for k in ("status","available_checkpoint_count","checkpoints_replayed",
    "checkpoints_rescored","checkpoints_matching_recorded_curve","registry_relative_path")},ensure_ascii=False))
assert result["status"]=="EVALUATION_COMPLETE"
assert result["available_checkpoint_count"]>0 and result["checkpoints_rescored"]>0
assert {row["id"] for row in result["inventory"]["routes"]}==set(plan["legacy_ids"])
assert all(row["status"]=="READY" and row["checkpoints"] for row in result["inventory"]["routes"])
assert all(cp["status"]=="READY" for row in result["inventory"]["routes"] for cp in row["checkpoints"])
assert result["checkpoints_rescored"]==result["available_checkpoint_count"]
'@ | & $py -X utf8 -B -
if ($LASTEXITCODE -ne 0) { throw 'E0 B 재추론 상태/curve mismatch/private log 확인' }
```

row128은 이 호출이 명시한 물리 F chunk이며 학습 조건을 바꾸지 않는다. 기존 immutable 평가가 다른 row128 identity 없이 이미 있으면 source/reference 계약에 따라 재사용 또는 거부될 수 있다. 실제 `executed_now`와 reused 수를 나눈다.

E0는 저장된 신규 cp>=2800와 연결된 selected old cp<=2400를 검사한다. endpoint/carry receipt, legacy4 source, exact A통계, 실제 foundation/state SHA를 검산한다. B만 target으로 쓰고 C origin은 비운다. 다만 압축 NPZ member와 전체 byte SHA가 물리적으로 행 격리되어 있다는 주장은 하지 않는다. E0 B는 과거 개발 자료이며 새 primary V/T 또는 독립 미관측 일반화 증거로 세지 않는다. `CURVE_SCORE_MISMATCH`, `BUDGET_PENDING`, inventory/route failure는 완성 수치에서 숨기지 않는다. E0 결과를 이미 잠근 E1–E4 선택에 되먹이지 않는다.

## 6. 최종 CPU 감사·보고

모든 GPU owner/child를 먼저 닫는다. CPU child에 CUDA=-1와 thread2를 명시하고 3단계의 `audit_campaign.audit_all` 명령을 다시 실행한다. `errors==[]`와 PASS_AVAILABLE_PREDICTIONS*를 실제 JSON에서 확인한다. CLI exit0만 보지 않는다. 선택 후 감사는 T manifest와 E0 legacy_B 독립 재채점까지 읽는다.

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& $py -X utf8 -B report_campaign.py --public-root . --private-root $env:INFORMS_CACHE_ROOT
if ($LASTEXITCODE -ne 0) { throw '보고서 생성 오류' }
```

실제 새 report의 SUMMARY/INPUT_ISSUES/INPUT_MANIFEST/REPORT_MANIFEST를 확인한다. 이 compiler는 저장 cost trials SHA·exact V/artifact binding과 V replay parity를 별도로 검산하므로 prediction audit만으로 비용 검산을 대신하지 않는다. report는 모델/F/timer를 다시 실행하지 않는다. 필수 범위가 남아 있으면 출력은 PARTIAL_SNAPSHOT이고 campaign_complete=false를 유지한다. 실제 audit의 mandatory_coverage_complete와 개별 단계의 실패/누락/재사용을 구분한다.

최소 결과 확인: 선택 ID별 T manifest/기준 SHA, V replay receipt PASS, 비용 B1/B8 fresh·balanced의 완료 범위, E0 actual/reused/curve mismatch, closed GPU charge 누계와 live owner0, 최종 audit errors0, report input issues를 확인한다. `report_campaign.RESUME_COMMANDS.ps1`는 V replay·명시 lock·E0 B 평가를 모두 실행하는 orchestration이 아니므로 그대로 실행해 이 절차를 대체하지 않는다.

게시 prepare/stage/commit/push는 이 runbook에 포함되지 않는다.

## 코드 근거

- `cost_campaign.py:143-188`: parser, 누적39600 ceiling, 자체 owner와 DEFERRED_RESOURCE 영수증.
- `cost_campaign.py:74-140`, `measure_costs.py:38-75,183-325`: 실제 조건·정확한 accuracy binding·warmup/timed/stage·fresh/balanced 구분.
- `resources.py:106-120,140-226,265-343`: 누적 process-wall, owner/child 인증, child join, deadline.
- `route_catalog.py:31-100,105-281`, `deployment.py:63-205,275-286,414-420`: receipt/artifact/parent/scaler/F identity와 실행/복구.
- `route_predictions.py:190-256`: replay의 원 V 위치 제한, 실제 subset과 tolerance, pre-inference seal.
- `final_evaluation.py:28-36,66-126,163-253,256-357`: source18, quiescence, 독립 감사 결박, immutable selection, 외부 audit owner, V/T 실제 API.
- `legacy_evaluation.py:228-351,360-470`, `legacy_continuation.py:245-254`: 실제 inventory/B prefix 평가 및 old source/statistics pins.
- `audit_campaign.py:277-435`: legacy namespace와 전체 예측 독립 감사, CLI의 JSON 결과를 직접 확인할 필요.
- `report_campaign.py:281-420,739-786,900-999`: cost/replay/legacy 저장 증거 재검산과 partial report, 기존 resume 명령의 범위.

