# 독립 코드 검토

검토 범위는 새 `common`, `run`, `worker`, `analysis`, `training`, `replay` 모듈과 대응 테스트다. 사전등록·실행 경계·재현 문서와 대조했다. 검토자는 소스와 기록을 읽었으며 파일 변경, 테스트 실행, GPU 작업, Git 게시를 수행하지 않았다. 실제 테스트·재현·실험 검산의 증거는 각각 별도 validation 파일에 있다.

현재 성공한 캠페인의 코드 게시를 막는 Critical 또는 Important 항목은 없었다. 실제 입력부터 CPU 출력까지의 측정, 전체 V와 ragged batch 검산, 비용만 사용하는 확인 선별, 네 검색의 공동 봉인 뒤 fresh A60 재학습, 공개 통계·그림 재생성 흐름을 확인했다. 이 코드 검토가 과학적 우월성이나 독립 일반화 검증을 대신하지는 않는다.

## 알려진 작은 제한

worker가 예외로 INVALID screen 기록을 만드는 경우 [worker.py](../../research/memory_budget_20261018_v1/worker.py)의 실패 summary에는 `dataset`과 `family`가 없다. [run.py](../../research/memory_budget_20261018_v1/run.py)의 `gate()`는 INVALID 처리보다 먼저 이 필드를 읽어 비용 추정치를 구성한다. 따라서 이 실패 경로에서는 의도한 `SKIPPED_GATE / INVALID_PHASE_A_COMPARISON` 봉인 대신 `KeyError`로 중단될 수 있다. 기존 완료 검사가 선택된 confirm의 미완료를 발견하면 그 검사가 먼저 중단시킬 수도 있다.

이번 Phase A의 screen 46개는 모두 COMPLETE·parity PASS였고 해당 분기를 거치지 않았다. 실제로 봉인한 PROCEED gate와 이후 Phase B의 결과는 이 제한의 영향을 받지 않는다. 측정 소스와 Phase B 생산자 정체성을 보존하기 위해 현재 캠페인의 소스는 변경하지 않았다.

향후 다른 등록에서 실패 복구를 개선할 때는 실패 summary에도 facet 축을 보존하거나 gate의 INVALID 검사를 비용 추정보다 먼저 수행하고, 실제 worker 실패 summary를 입력하는 직접 `run.gate()` 회귀 검증을 추가한다. 이번 캠페인의 봉인을 삭제하거나 변경해 우회하지 않는다. 이 항목은 완료된 과학 실험의 미측정 facet을 뜻하지 않는다.

## 검토 범위의 경계

- 기존 runtime 전체를 다시 감사하지 않고 새 호출과 직접 관련된 경로를 확인했다. 기존 API와 데이터 경계의 별도 감사 기록을 함께 사용한다.
- 검토 시점의 코드 판단과 최종 실험·테스트·원격 검증의 완료 증거를 구분한다.
- 부정확한 최고 성능·전체 프로세스 hard cap·미측정 chunk의 전역 최적성을 주장하는 근거로 사용하지 않는다.
