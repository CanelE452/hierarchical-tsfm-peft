# 일반 whole-path 비용의 별도 수치 검산 준비

[audit_core_costs.py](audit_core_costs.py)는 ROOT가 지정한 최종 report의 `COSTS.csv`·`INPUT_MANIFEST.json`·`REPORT_MANIFEST.json`을 정확한 SHA로 고정한 뒤, 그 보고서가 참조한 receipt/trial gzip만 읽는 stdlib CPU 감사기다. 준비 상태에서는 실제 report·cost receipt·gzip을 읽지 않는다. 모델·NPZ·weights·원자료·GPU·새 timer 측정은 실행하지 않는다.

기존 `report_campaign._costs()`가 timed median/mean·phase count와 정확한 route binding을 검증한 것과 별도로, 저장 p90·stage·balanced 순서·F counter를 검산한다. 실제 report가 완료되고 ROOT가 명시적으로 지시하기 전에는 `--execute`를 실행하지 않는다. K20의 별도 전환 형식은 [분리 감사](../multik_numeric_audit_20261016_v1/README_KO.md)를 사용하며 이 helper에 섞지 않는다.

## 고정 규약

원 `measure_costs.summarize`와 `measure_balanced` 전체를 읽었다. single worker도 같은 measure_balanced에 route 하나를 전달한다. phase는 warmup→timed→stage_probe이며 full protocol은 각각 최소10/20/20이다. balanced의 route 순서는 receipt의 `run_ids`를 사용하고 repeat r·position p에서 `run_ids[(r+p)%N]`이다. run ID 정렬로 이 순서를 대체하지 않는다. declared partial protocol은 실제 상태로 별도 보존한다.

검산 범위는 다음과 같다.

- 명시된 report SHA와 REPORT_MANIFEST의 output SHA/byte 수를 대조하고 INPUT_MANIFEST의 public receipt·gzip SHA/byte 수를 실제 stream hash로 확인한다. gzip을 CSV 끝까지 읽어 CRC도 검사한다. 비용 전체 폴더를 glob하지 않는다.
- condition별 accuracy/cost spec의 복합 fingerprint·선언 ID·허용 cost-only 필드·data/scaler와 report의 V/artifact SHA metadata를 대조한다. private artifact의 INPUT_MANIFEST 기록을 비교하며 모델 파일은 열지 않는다. report가 이미 수행한 실제 모델 byte 검산을 새로 했다고 쓰지 않는다.
- warmup/timed/stage의 trial·position·condition 순서, stage_instrumentation과 비계측 stage 빈 칸, whole/stage finite 비음수 시간, row microbatch와 F logical/physical streams·actual calls를 검증한다. top-level row128을 RAW 부모에 일괄 적용하지 않고 실제 spec의 F leaf별 rowmicrobatch를 재귀 계산한다. `past_control`은 b 부모를 호출하지 않는다.
- timed와 intrusive stage를 구분해 저장 median/mean/min/max/p90를 재계산한다. p90는 정렬된 값의 `(n−1)×0.9` 위치에서 선형 보간하며 NumPy를 import하지 않는다. 평균은 `math.fsum(values)/n`이다. 비교는 ROOT가 지정한 고정 absolute1e−9·relative1e−7 안에서만 수행하고 실행 중 오차를 늘리지 않는다. 이는 집계 산술 비교이며 원 예측 parity tolerance를 변경하지 않는다.
- source39·registry·selection lock과 report·읽은 통계의 SHA를 전후 검사한다. 한 receipt의 오류는 FAILED_PRESERVED로 남기고 독립 receipt를 계속 검산한다. 원 core72의 실패12건에 가짜 시간·메모리 값을 만들지 않는다. 이 감사의 coverage는 **명시 report에서 실제 참조한 receipt**이며, 미실행/실패 scope 전체의 완료와 구별한다. 기존 PeMS LoRA fresh2가 그 report에 포함되면 같은 방식으로 검산한다.

whole 시간은 물리 입력 검증·전처리·전송·예측·FP64 역변환·CPU 출력·sync를 포함한다. 바깥 route activate/deactivate와 로딩은 이 타이머 밖이며 balanced LoRA 교체 비용을 포함했다고 해석하지 않는다. stage_probe는 추가 sync를 넣은 intrusive 측정이다. stage 합=whole은 강제하지 않는다. memory는 원 receipt 기록이며 새로 재측정하지 않는다. CUDA peak는 timed 시작 reset 이후 stage_probe까지 포함하고 RSS after close는 RSS peak가 아니다. 저장 prediction SHA를 배열에서 다시 계산한 감사도 아니다.

## 명령 형식

```powershell
$Recipe = Join-Path $env:INFORMS_PUBLIC_ROOT 'diagnostics/core_cost_numeric_audit_20261016_v1'
& $Python -X utf8 -B (Join-Path $Recipe 'audit_core_costs.py')
& $Python -X utf8 -B (Join-Path $Recipe 'test_audit_core_costs.py')

# 실제 report terminal과 ROOT 지시 이후에만 실행
& $Python -X utf8 -B (Join-Path $Recipe 'audit_core_costs.py') --execute --root-triggered `
  --report-relative '<최종 actual report의 campaign 상대경로>' `
  --costs-sha256 '<COSTS.csv SHA>' --inputs-sha256 '<INPUT_MANIFEST.json SHA>' `
  --report-manifest-sha256 '<REPORT_MANIFEST.json SHA>' `
  --output 'diagnostics/core_cost_numeric_audit_20261016_v1/audit_<새 고유 ID>'
```

출력은 `AUDIT_RESULT.json`, `FILE_BINDINGS.json`, `SUMMARY_RECALCULATIONS.csv`다. 작은 합성 검사는 fresh/balanced·linear p90·고정 산술 오차·회전/stage 변조·nested RAW의 F calls·metadata binding·음수 timer를 확인한다. 준비 검사 통과는 실제 core 비용 전수 감사 완료 또는 비용 우위의 증명이 아니다.
