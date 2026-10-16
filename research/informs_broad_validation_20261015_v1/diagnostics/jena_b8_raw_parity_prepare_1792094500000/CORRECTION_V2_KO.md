# 준비 helper v2의 표현 교정

[확인] 원 helper와 기존 영수증을 보존하고 `capture_original_parity_v2.py`를 새 파일로 만들었다. 실제 실행에는 v2를 사용한다. 원 parity 호출·기본 tolerance·cohort 구성·모델·소스·진단 결과 판정은 바뀌지 않았다. 바뀐 것은 INPUT_BINDING의 관측 범위 표현 하나뿐이다.

```text
기존: target_values_read=False
교정: forecast_target_arrays_constructed_or_scored=False
      parity_forward_inputs_are_V_past_histories=True
      whole_asset_file_hash_or_load_is_not_row_isolated=True
```

`load_task`는 전체 허용 safe-FIT 배열/파일을 읽을 수 있다. 따라서 원 표현을 파일 바이트 전체를 읽지 않았다는 주장으로 해석하면 안 된다. 진단이 실제 예측 입력으로 사용하는 것은 V 과거 창이며 미래 목표 배열을 만들어 채점하지 않는다는 범위이다. 교정 블록을 역치환하면 원 source text와 정확히 같다. v2 AST 검사와 기본 metadata CLI가 통과했다. 기존 토이 테스트2개는 원 helper에 수행한 근거이며 v2에서 다시 수행했다고 세지 않았다. 실제 GPU 진단은 아직 0이다.

# Electricity B1 별도 stderr 확인

[확인] `costs/electricity_1792094083565791200`의 `fresh_b1_m000_rowsfull/32/8/4`도 모두 exit1/성공 영수증 없음이다. 각 private stderr의 마지막 예외는 Jena와 같은 `measure_costs.py:207`의 `Actual Raw row variants disagree or all failed`였다. 각각 약 13.43~13.83초 후 종료했다.

[확인] 이는 같은 gate 위치에서 실패했다는 근거다. 어느 variant가 OOM 또는 tolerance 초과였는지는 여기서도 반환값이 사라져 구별할 수 없다. Jena B8과 동일한 수치 원인이라고 단정하지 않는다. Electricity B1도 실패했으므로 현재 문제를 Jena 또는 B8에만 생기는 것으로 한정할 근거도 부족하다. 추가 Electricity 진단 구현·GPU 실행은 하지 않았다. 원 stderr 전체는 기계 경로를 포함하므로 공개하지 않고 `ELECTRICITY_B1_METADATA_REVIEW.json`에 안전한 예외 요약과 원 SHA만 남겼다.
