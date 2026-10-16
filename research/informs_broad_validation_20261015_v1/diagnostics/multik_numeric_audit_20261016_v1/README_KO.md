# K 전환 비용의 독립 숫자 검산 준비

[audit_multik.py](audit_multik.py)는 저장된 비용 JSON/CSV만 읽는 stdlib CPU 검산기다. 새 추론·학습·timer·메모리 측정이 아니며 모델·체크포인트·NPZ·원자료·GPU를 읽거나 실행하지 않는다. 현재 비용 폴더 접근과 실제 K20 검산은 아직 하지 않았다. 기본 실행은 고정 pins·원 classifier 소스의 정적 메타데이터만 확인한다.

기존 `report_campaign`의 core 비용 집계는 K20의 특수 RESULT/NUMERIC/전환 ledger를 자동 포함하지 않는다. 이 검산은 별도 `AUDIT_RESULT.json`, `FILE_BINDINGS.json`, `TRANSITION_SUMMARIES.csv`로 남기며 기존 core 비용과 합치거나 기존 원본을 수정하지 않는다.

## 원 규약과 검산 범위

원 launcher `classify_scope`, `_resume`, `run_with_owner` 전체와 원 writer의 schedule/ledger/summaries를 읽었다. 원 launcher의 공개 byte 동일 사본 SHA는 `fe629d2586165984f3fe980c19340a1e15aba84aa6fc87a81482c09e849bb7b3`이다. 감사는 해당 코드를 변경하거나 GPU 경로를 호출하지 않고 독립적으로 저장 수치를 계산한다. 준비 시 schema 확인에는 이미 완료된 K8의 Beijing B1/shared 한 표본만 사용했다.

- ROOT가 비용 owner/process terminal을 확인한 뒤 `--execute --root-triggered`로만 실행한다. 정확한 최종 BATCH·원 pins·ROOT terminal RESULT의 SHA가 명시 입력이다. 실제 중앙 CLOSED 영수증 SHA/필드와 ACTIVE 부재를 확인한다. 전체20 scope 순서·route/K/seed·원 wall deadline·등록 scientific contract·고정 V manifest SHA를 대조한다.
- 최종 BATCH의 한 scope당 최종 참조 하나만 읽는다. 재사용8과 fresh 조건을 합쳐 최대20이며, 이전 별도 partial166을 중복 합산하지 않는다. 원 queue·상태·실패를 바꾸지 않는다.
- scope의 모든 원 파일 경로/SHA, RESULT/NUMERIC rows·ledger, CSV의 순서/문자열 값, 각 trial의 원 Windows CRLF JSON 바이트와 연속 번호를 비교한다. 세 K의6 directed edge를 repeat마다 회전한 정확한 schedule prefix를 독립 재생성한다. warmup10/timed20 요청과 실제/잔여/resource-valid 수를 재계산한다.
- 저장 summary는 **timed의 전체 저장 trial** median이다. `statistics.median`으로 두 원 필드를 재계산해 정확 비교한다. 별도 CSV의 `all_recorded`와 `resource_valid_only`를 구분한다. partial에서 원 summary가 없으면 그 사실을 남기고 실제 저장 trial의 계산값만 별도 제시한다. p90은 계산하지 않는다.
- 실제 duration은 finite·0 이상을 확인한다. `other_clock_overhead_seconds`는 차감 잔차여서 finite signed 값으로 보존한다. guard 합과 derived 값은 원 writer의 동일한 연산 순서로 exact 검산한다. `physical_pipeline_seconds`와 whole은 다른 계측 범위이며, 원 writer의 `substage_sum_equals_whole_claimed=False`를 유지한다. 새로운 prediction tolerance를 도입하지 않는다.
- memory/storage는 저장 byte 필드의 값·단위·합계와 고정 route artifact metadata를 검증한다. 현재 RSS는 RSS peak가 아니며 CUDA peak는 원 measurement reset 이후 값이다. `foundation_instances`는 생성 횟수다. 과거 메모리나 모델 byte를 새로 측정·로드·재해시한 것으로 쓰지 않는다.
- NUMERIC의 pending/complete=false는 의도적인 중간 봉인 상태다. 최종 RESULT와 rows/ledger는 같아야 하지만 상태 필드까지 같다고 요구하지 않는다. child protocol, global owner resource validity와 저장 수치 검산의 PASS를 구분한다. guard가 포함된 instrumented 비용을 순수 배포 latency로 해석하지 않는다.
- source39·registry·selection lock을 전후 byte 동일 검사하고 모든 읽은 통계의 SHA도 다시 비교한다. 한 scope의 오류는 해당 `FAILED_PRESERVED`로 남기고 독립 scope 검산을 계속한다. 원본은 덮거나 삭제하지 않는다.

## 실행 명령 형식

아래 actual 명령은 아직 실행하지 않았다. ROOT가 terminal RESULT와 최종 BATCH SHA를 확정한 뒤 새 고유 output을 지정한다.

```powershell
$Recipe = Join-Path $env:INFORMS_PUBLIC_ROOT 'diagnostics/multik_numeric_audit_20261016_v1'
& $Python -X utf8 -B (Join-Path $Recipe 'audit_multik.py')
& $Python -X utf8 -B (Join-Path $Recipe 'test_audit_multik.py')

# ROOT의 실제 비용 terminal 확인과 명시 지시 이후에만
& $Python -X utf8 -B (Join-Path $Recipe 'audit_multik.py') --execute --root-triggered `
  --batch-relative 'costs/<최종 실제 batch>/BATCH_LAUNCHER_RECEIPT.json' `
  --batch-sha256 '<최종 BATCH SHA>' `
  --pins-sha256 'b520924528606a1f2eb83f6302a247cfd9b1f4e5a179adb2f1ff83b5fc3c2ad0' `
  --terminal-result 'diagnostics/<실제 ROOT terminal>/RESULT.json' `
  --terminal-sha256 '<실제 terminal RESULT SHA>' `
  --output 'diagnostics/multik_numeric_audit_20261016_v1/audit_<새 고유 ID>'
```

## 현재 증거의 한계

[test_audit_multik.py](test_audit_multik.py)는 작은 합성 자료로 완전180·부분166/유효165, 정확 median, 방향 순서, trial 바이트/CSV 손상, signed 잔차와 음수 duration, guard/derived 산술 오류를 검사한다. 기본 metadata 실행은 framework를 import하지 않고 현재 비용을 읽지 않는다. 이 준비 검사 통과를 실제 K20 전수 검산 완료나 비용 우위로 바꾸어 말하지 않는다. 실제 원 예측 SHA는 배열을 재추론해 확인한 것이 아니며 역사적 model/source binding과 새 독립 수치 집계의 범위를 구분한다.
