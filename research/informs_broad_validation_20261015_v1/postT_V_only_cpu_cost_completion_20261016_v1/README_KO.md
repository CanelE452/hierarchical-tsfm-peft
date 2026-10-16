# 새 required6·family4의 CPU 전체 파이프라인 비용 준비

**[확인] 이 디렉터리는 10개 조건의 B1/B8 비용 작업을 준비하는 코드와 tiny 시험이다. 실제 비용 20조건을 측정했다는 결과가 아니다.** ROOT가 각 학습·full-V·독립 FP64·실제 fresh CPU6 replay 완료를 확인하고 명시적으로 실행해야 한다. 이 helper는 학습·추론·비용 worker를 시작하거나 종료하지 않는다. 다른 조건이 미완료여도 유효한 한 조건을 독립적으로 준비할 수 있다.

[준비 코드](prepare_completion_costs.py) · [tiny 시험](test_prepare_completion_costs.py) · [required6 PLAN](../postT_V_only_cpu_required_minima_20261016_v1/PLAN.json) · [family4 PLAN](../postT_V_only_cpu_family_minima_20261016_v1/PLAN.json)

## 범위와 실제 API

원 [route_catalog.py](../route_catalog.py)의 `build_route(run_id, task, cache_root, public_root)`가 만드는 selected fit·scaler·foundation·standard V canonical spec을 그대로 사용한다. 원 [measure_costs.py](../measure_costs.py)의 `--worker`를 실행할 명세만 생성하고 이 파일을 복사하거나 수정하지 않는다. 기본 호출은 stdlib만 사용하며 메타데이터 파일 읽기·쓰기·Task·모델·배열·child 작업이 모두0이다. `--inspect`는 명시된 원본/자산 root의 동결 PLAN·source39·원 불변자료9·정규화 source·registry alias를 검사하지만 완료 fit·V·replay 또는 비용의 증거가 아니다.

`--execute`는 한 exact kind/ID/PLAN SHA, 실제 FP64 rescore RESULT의 경로·SHA, 실제 fresh CPU6 replay RESULT의 경로·SHA, 살아 있는 deadline과 새 output을 모두 요구한다. 기존 verifier의 `prepare_inputs`를 수정 없이 호출하여 complete3epoch search·선택된 fresh A60 refit·원 recipe/정규화/초기화·선택 artifact·full V43/H48/C와 파일 SHA를 대조한다. 원 [required verifier](../postT_V_only_cpu_required_minima_20261016_v1/verify_required_minima_outputs.py)와 [family verifier](../postT_V_only_cpu_family_minima_20261016_v1/verify_family_minima_outputs.py)의 바인딩 규약을 재사용한다.

FP64 RESULT의 과학적 범위와 현재 fit/V 입력 hash가 같아야 한다. CPU replay는 단순 `REPLAY_RECEIPT.json` 존재로 인정하지 않는다. 원 verifier source SHA·PLAN SHA·requested/actual CPU2·CUDA false·실제 forward true·`scientific_result.executed_now=true`·단일 batch6·고정 tolerance·현재 artifact/manifest/reference/archive·원 source/route fingerprint와 `REPLAY_BINDINGS.json`을 함께 검사한다. cache-hit replay나 과거 batch4의 존재는 fresh CPU6 조건을 충족하지 않는다. 이 검사는 선행 실행의 정확한 바인딩을 재확인하며 준비 명령이 FP64 재채점이나 replay를 다시 실행하지는 않는다.

준비의 실제 수치 접근은 원 safeFIT Task 로드와 selected adapter의 원 catalog 로드다. foundation 모델 생성·예측·optimizer·timer는0이다. 원 safeFIT 로더가 과거 campaign-T bytes까지 물리 역직렬화할 수 있다는 한계를 보존하며, 세션 전체 T-byte 접근0이라고 주장하지 않는다. 새 T target 생성·평가·선택 API는 호출하지 않는다.

## 정확한 조건과 K

```text
kind=required; PLAN SHA=04acc4aabffd02a7266f16fee5c6a3884313fac6ca4b5b46ef0db45eb2fe5152
postT_cpu_beijing_E1_B2_k6_s92601_required_v1
postT_cpu_beijing_E3_LAG_CAUSAL_k6_s92601_required_v1
postT_cpu_beijing_E3_LAG_BIDIR_k6_s92601_required_v1
postT_cpu_beijing_E1_B2_k11_s92601_required_v1
postT_cpu_beijing_E3_LAG_CAUSAL_k11_s92601_required_v1
postT_cpu_beijing_E3_LAG_BIDIR_k11_s92601_required_v1

kind=family; PLAN SHA=e34033935916dcaded0885ccde8e1aa084f5cc0df994024d0827510d6db53b10
postT_cpu_jena_E1_B4_k6_s92601_family_v1
postT_cpu_jena_E1_ADAPTS_BOLT_PORT_k6_s92601_family_v1
postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1
postT_cpu_beijing_E1_B4_k11_s92601_family_v1
```

required는 Beijing task 하나와 실제 K6/11을 사용한다. family는 Beijing/Jena의 서로 다른 scaler와 C11/21을 구분한다. LoRA는 등록 `LORA_R8/K0`와 실제 `family=LORA/F width21`이 다르며, AdaPTS 원리 포트도 등록 `ADAPTS_BOLT_PORT`와 실제 `ADAPTS`를 구분한다. 한 조건당2개 명세, 10개가 각각 성공할 때만 합20개 명세다. 명세 생성20개는 실제 비용20개와 다르다.

## ROOT 실행 순서

아래는 아직 실행하지 않은 형태다. 변수에는 ROOT가 확인한 실제 root·완료 영수증·새 output을 넣는다. script 이름만 실행하면0read 설명만 출력한다.

```powershell
& $python -X utf8 -u -B "$campaign/postT_V_only_cpu_cost_completion_20261016_v1/prepare_completion_costs.py"

# 원본 root의 source/PLAN 검사만 수행하며 모델/Task/배열/worker는 읽거나 실행하지 않는다.
& $python -X utf8 -u -B "$campaign/postT_V_only_cpu_cost_completion_20261016_v1/prepare_completion_costs.py" `
  --inspect --public-root $campaign --cache-root $campaignCache --asset-root $assetRoot `
  --kind $kind --run-id $runId --plan-sha $planSha

# ROOT의 명시 trigger 이후에만 수행: 완료된 한 조건의 B1/B8 spec을 준비한다.
$namespace = if ($kind -eq 'required') { 'postT_V_only_cpu_required_minima_20261016_v1' } `
  elseif ($kind -eq 'family') { 'postT_V_only_cpu_family_minima_20261016_v1' } else { throw 'Unknown kind' }
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:INFORMS_ASSET_ROOT = $assetRoot
$env:INFORMS_CACHE_ROOT = Join-Path $campaignCache $namespace
& $python -X utf8 -u -B "$campaign/postT_V_only_cpu_cost_completion_20261016_v1/prepare_completion_costs.py" `
  --execute --public-root $campaign --cache-root $campaignCache --asset-root $assetRoot `
  --kind $kind --run-id $runId --plan-sha $planSha `
  --rescore-result $actualRescoreResult --rescore-result-sha $actualRescoreSha `
  --replay-result $actualFreshCpu6Result --replay-result-sha $actualFreshCpu6Sha `
  --deadline $liveDeadline --output $freshPreparationOutput
```

준비 환경은 기존 runner와 같다: `CUDA_VISIBLE_DEVICES=-1`, OMP/MKL/OPENBLAS threads2, HF/Transformers offline1, `INFORMS_ASSET_ROOT=$assetRoot`. `INFORMS_CACHE_ROOT`는 kind별 **selected fit namespace**다: required이면 `$campaignCache/postT_V_only_cpu_required_minima_20261016_v1`, family이면 `$campaignCache/postT_V_only_cpu_family_minima_20261016_v1`. CLI `--cache-root`에는 이와 구분하여 **원 campaign cache root `$campaignCache`**를 넘긴다. 원 namespace 환경 guard는 수정하지 않는다. 준비 helper output은 이 별도 cost-completion public namespace 안의 새 하위 폴더여야 한다. 기존 output을 덮지 않는다.

`COST_EXECUTION_PLAN.json`의 `status=PREPARED_NOT_EXECUTED_CPU_COST_WORKERS`와 입력 보존 true인 명세만 ROOT가 검토한다. FAILED/부분 output은 보존하며 그 안의 job을 자동 실행하지 않는다. `--execute`는 어떤 worker도 자동으로 실행하지 않는다. 각 job의 인자는 다음 placeholder를 가진다.

```text
{PYTHON}                검증된 Python executable
{ORIGINAL_PUBLIC_ROOT} 원 campaign public root
{ORIGINAL_CACHE_ROOT}  원 campaign private cache root
{POST_T_PUBLIC_ROOT}   원 public root / job.worker_public_relative_root
{POST_T_CACHE_ROOT}    원 cache root / job.worker_cache_relative_root
{ASSET_ROOT}           실제 기존 자산 root
```

spec은 원 cache의 이 cost-completion namespace에만 저장한다. worker의 public/cache root는 해당 selected fit·V가 있는 required 또는 family namespace다. 미래 비용 출력은 원 public root의 이 cost-completion namespace로 분리된다. 원 registry·SCORES·COST·selection lock·T 결과는 수정하지 않는다.

## 실제 측정과 별도로 남는 조건

ROOT가 살아 있는 모든 학습·추론·다른 비용 작업의 종료와 실제 process identity를 확인한 quiet window에서만 **한 번에 한 fresh CPU worker**를 실행한다. helper에 쓰인 quiet 요구 문구나 파일 존재가 quiet 상태를 증명하지 않는다. 원 worker에는 CPU deadline/RSS 제한이 내장돼 있지 않으므로 ROOT의 외부 process-tree supervisor가 필요하다. CPU2·RSS3GiB·deadline1792222550 안의 더 이른 명시 deadline·합계 저장30GiB·여유256MiB를 지킨다. helper는 준비 과정 자체의 RSS/deadline watchdog을 사용하며 다른 프로세스를 종료하지 않는다.

원 비용 범위는 CPU 물리 입력→관측 mask 검사/과거 채움·정규화→P/E/F/D→FP64 역변환→CPU 반환/동기화다. model/route/foundation loading과 steady-state는 따로 기록하며 Python/data startup 또는 cold disk cache 비용으로 확대하지 않는다. B1/B8 각각 warmup10·timed20·stage20이고, stage probe는 별도 intrusive scope다. F logical streams·physical streams·calls는 실제 원 trial telemetry를 확인하며 full V bundle의 bookkeeping F_calls=1을 실측 비용 호출 수로 쓰지 않는다.

각 worker는 fresh single-route scope다. 이20개는 shared-process balanced 비교를 대신하지 않는다. 원 worker는 training peak를 측정하지 않는다(`training_peak_included=false`). CPU resident weight accounting/after-close RSS와 외부 supervisor의 CPU peak RSS도 구분한다. GPU-only active-adapter 필드를 CPU activation memory라고 부르지 않는다.

LoRA에서는 원 deployment `activate()`가 실제 merge하고 worker의 `check_lora_parity()`가 unmerged/merged 예측을 고정 atol1e-5/rtol1e-4로 검사한다. `close()`의 unmerge/adapter close/FoundationPool frozen hash 확인을 그대로 사용한다. 향후 실제 영수증의 parity PASS와 exact frozen restore를 확인하기 전에는 병합비용 측정 완료라고 쓰지 않는다. 숫자·F counter·trial 수·source/artifact binding의 사후 독립 검산도 별도 작업이다.

## 이 구현에서 실제 실행한 검증

[tiny 시험](test_prepare_completion_costs.py) **14개 PASS**를 명시 실행했다. 첫 구현 부재에서8개 실패를 확인했고, 이후 cached scientific replay를 fresh로 받아들이는 추가 시험을 실제 실패시킨 뒤 수정했다. 마지막 실행은14개/3.847초다. required 및 family4 모두 원 verifier의 작은 fixture와 실제 검증 함수를 재사용한다. opaque `.pt`/`.npz` 문자열 바이트를 hash할 뿐 실제 Task·모델·배열은 역직렬화하지 않는다.

기본0read·source 변조 import 전 거부·현재 fit/V/CPU6 proof 대사·canonical route 변조 거부·LoRA 등록/실제K 구분·다른 미완료 조건 비차단·정확한 worker argv·fake runtime의 spec2개/비용0·기존 output 보존·틀린 proof 전 과학 호출0·catalog 중 입력 변조 시 job 발행0을 검사했다. fake runtime 성공은 실제 cost 준비 실행이나 비용 측정의 증거가 아니다. 이 subtask에서는 실제 `--inspect`/`--execute`·Task·추론·비용·subprocess·Git를 실행하지 않았다. ROOT가 실제 선행 결과와 자원 상태를 확인해 다음 단계를 수행한다.
