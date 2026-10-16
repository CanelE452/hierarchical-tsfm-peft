# 원 parity 반환 보존을 위한 추가 진단 준비

[확인] 현재는 준비 상태다. 실제 GPU 진단은 아직 실행하지 않았다. 원 함수의 반환에 variant별 OOM/PASS/FAIL·max_abs·허용오차가 들어 있으므로 추가 출력 비교나 임계 완화는 만들지 않았다.

`capture_original_parity.py`는 원 `measure_costs.raw_row_parity()`를 기본 candidates/tolerance 그대로 딱 한 번 호출하고 반환 즉시 `ORIGINAL_PARITY_RETURN.json`을 새 파일로 보존한다. 이후 close나 사후 경계 검사가 실패해도 이 파일은 남는다. raw forecast·targets는 공개하지 않는다. 원 worker의 실패 기록은 보존한다.

진단은 원 Jena B8 full spec bytes, 첫 8개 V stride-H의 과거 창 생성, 데이터/scaler, 실제 foundation state, 원 model revision, source39 및 외부 instance-norm source를 결박한다. 원 B8 실패는 입력 hash를 저장하지 못했으므로 존재하지 않는 원 입력 hash와의 비교를 주장하지 않는다. 같은 고정 spec/data와 원 worker의 동일한 cohort 생성 코드로 실제 진단 입력 hash를 새로 기록한다. Raw에는 별도 adapter artifact가 없으며 정확한 foundation hash가 필요하다.

기본 CLI는 작은 source/spec/receipt/log/instance-norm 파일의 해시만 확인한다. NumPy·torch·과학 모듈을 import하거나 모델/배열을 읽지 않는다. `$R`은 공개 연구 루트, `$C`는 비공개 캐시, `$A`는 기존 asset 프로젝트, `$PY`는 프로젝트 Python이다.

```powershell
& $PY -X utf8 -u -B "$R/diagnostics/jena_b8_raw_parity_prepare_1792094500000/capture_original_parity.py" --public-root $R --cache-root $C --asset-root $A
```

실제 GPU 실행은 ROOT의 별도 실행 지시 후 기존의 같은 PID audit owner에서 `run_with_owner(lease, asset_root, cache_root, public_root)`를 한 번 호출한다. 이 helper는 owner를 생성·갱신·종료하지 않고 기존 인증 child 하나를 새로 시작해 종료를 기다린다. owner 누적 GPU 상한 39600초와 최종 3600초 보존, 기존 owner의 실제 deadline을 유지한다. 인증이 없으면 모델·입력 처리 전에 거부된다. 정밀 비용 실행 중 지금 호출하면 안 된다.

반환 JSON을 보존한 진단의 exit0은 원 cost protocol 성공을 뜻하지 않는다. `original_gate_passed=false`일 수 있으며 warmup/timed/stage trial은 모두 0, `cost_protocol_completed=false`로 기록한다. 원 parity 결과가 실패하면 먼저 그 수치/상태를 해석해야 한다. 원 gate를 우회하거나 다른 허용오차로 성공을 대체하지 않는다.

준비 검증은 AST2개와 토이 JSON 보존 테스트2개만 실제 실행했다(exit0). 이 테스트는 원 GPU 예측을 검증하지 않는다. 기본 metadata CLI도 exit0, NumPy/torch import=false였다. 실제 모델·배열·GPU 진단 호출은 0이다. 최종 실제 진단 결과를 게시할 때는 새 출력의 경로 안전성도 다시 확인해야 한다.
