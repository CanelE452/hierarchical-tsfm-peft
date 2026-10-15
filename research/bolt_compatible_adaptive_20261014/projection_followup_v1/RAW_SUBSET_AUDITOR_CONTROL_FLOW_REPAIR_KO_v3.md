# 원 실험을 보존한 독립 감사 제어 흐름 복구

상태: 감사 v2 실제 실패 후 새 v3 감사 실행 전 기록이다. 원 scientific producer v2는 COMPLETE지만 아직 독립 PASS가 아니다. 비용/최종 보고의 PASS gate를 건너뛰지 않는다.

[확인] 실제 `runs/raw_subset_comparison_v2/RESULT.json` SHA `2b271ba9aa6ae359033444b282ddc1becc5d1f60ef063451171e9c93f70db4ff`를 감사한 `audit_raw_subset_comparison_v2.py` SHA `2fcd351ba4a1e4982fce1bb50d3fe34e4ec5b5f934850e902f10c84c8695acd6`는 `The truth value of an array with more than one element is ambiguous`로 FAILED_PRESERVED했다. [원 실패](audits/RAW_SUBSET_COMPARISON_AUDIT_v2.json)·[같은 감사 source의 추적 실행 실패](audits/RAW_SUBSET_AUDIT_FAILURE_TRACE_REEXECUTION_v1.json)·[ROOT가 잡은 실제 stack](audits/RAW_SUBSET_AUDIT_EXCEPTION_TRACE_v1.json)을 보존한다.

[확인] 실제 exception은 audit_result의579행 `if complete and not report["audit_failed_paths"]:`다. 이 함수에서 complete는 처음 producer 상태를 나타내는 bool이지만, 아래 E(Y) 진단에서 `context["observed"].all(axis=2, keepdims=True)`라는 관측 mask ndarray로 재할당된다. 마지막 집계가 그 배열을 bool로 판단하면서 실패했다. 새 v3는 producer 완료 여부와 complete-C target mask의 변수 이름만 분리한다.

이 복구는 모델·원 scientific source·greedy·global D·학습·checkpoint·예측 배열·선정·수식·공차를 바꾸지 않는다. 원 실험을 다시 학습하거나 F를 재추론하지 않는다. 검산 source와 실패 결과도 덮지 않으며 새 source/출력에서 같은 독립 LS·SVD·실제 저장 CP 추론·점수·선정·출력 공간·원파일 보존 검사를 실행한다. ROOT가 새 source diff와 합성 회귀 검사를 확인한 뒤 실제 감사를 호출한다. 실패 artifact가 생성됐다는 사실을 과학적 검산 통과로 세지 않는다.

실패와 추적에 든 CPU 사용량도 보존한다. 최신 상한 해제 지시로 기존 시간 상한 때문에 멈추지 않지만 TEST·설치·큰 백본 학습·기존 작업 종료/중단 실험 재개·자동 commit/push는 하지 않는다. v3의 실제 PASS와 원수치가 생기기 전에는 본 문서가 완료나 개선을 보증하지 않는다.
