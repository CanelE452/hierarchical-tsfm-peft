# Jena Raw B8 실패 원인: 확인된 범위

[확인] `costs/jena_1792093667627587500`의 `fresh_b8_m000_rowsfull/32/8/4`는 모두 child exit1이며 성공 영수증이 없다. 네 stderr의 마지막 원인은 `measure_costs.py:207`의 `Actual Raw row variants disagree or all failed`이다. 각각 약 11.0~11.2초 후 종료했다. 원 로그와 실행 영수증은 수정하지 않았다.

[확인] 원 worker는 요청 row variant 하나만 검사하지 않는다. `raw_row_parity()`가 같은 입력에서 full/32/8/4를 모두 비교하고, 그 반환의 `successful_variants_parity`가 false면 timed protocol 이전에 종료한다. 따라서 네 번의 동일 공통 검사 실패를 독립된 네 종류의 원인으로 세면 안 된다. warmup10/timed20/stage20은 시작하지 않았다.

[확인] false의 가능한 이유는 (1) 네 variant가 모두 CUDA OOM으로 실패해 기준 출력이 없는 경우, 또는 (2) 성공한 variant 사이에 atol=1e-5, rtol=1e-4를 넘는 출력 불일치가 있는 경우이다. OOM은 원 함수 내부에서 잡혀 반환 rows에만 기록되므로 stderr에 OOM 문구가 없다는 사실로 배제할 수 없다. 실패 worker는 반환값을 저장하기 전에 throw하여 현재 기록만으로 어느 경우인지 구분할 수 없다. 특정 채널·좌표·오차 크기 또는 학습 문제로 원인을 단정할 근거도 없다.

[확인] B1은 네 variant가 모두 통과했다. physical max_abs는 full/32=0, rows8=0.00013508011761587113, rows4=0.00016885014701983891이었다. B1과 B8의 variant별 spec JSON bytes는 동일하다. 원 source에서 B1은 첫 1개, B8은 첫 8개 V 과거 창을 사용하고, Jena의 C=21이므로 논리 F stream 수는 21과 168이다. 입력 history와 물리 row batch 조건이 다르므로 B1 통과를 B8 통과로 재사용할 수 없다.

[추정] batch·cohort에 따른 수치 차이가 후보이지만 OOM이나 구현 원인을 현재 근거로 확정하지 않는다. 동일 네 실패를 그대로 반복하기보다 원 parity 반환 한 번을 보존하는 진단이 먼저 필요하다. 원 gate·허용오차·과학 소스는 유지한다.

[확인] `cost_campaign.py:119~122`는 성공 fresh 조건만 balanced 후보에 넣는다. Raw B8이 전부 실패하면 balanced 실행이 다른 방법으로 진행되더라도 Raw를 포함한 전체 비용 비교로 해석하면 안 된다. 해당 경로의 미측정 상태를 남기고 독립 가능한 다른 비용 조건은 계속할 수 있다.

근거는 원 worker `measure_costs.py:91~120,185~224`, 각 `fresh_b8_m000_rows*.execution.json` 및 비공개 동명 `.log`, B1의 `fresh_b1_m000_rowsfull.json`이다. 계획 파일에 원본 SHA를 고정했다. 비공개 stderr에는 기계 경로가 있으므로 이 문서에는 원문 전체를 복사하지 않는다.
