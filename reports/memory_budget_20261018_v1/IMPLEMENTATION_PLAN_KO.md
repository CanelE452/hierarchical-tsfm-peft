# Memory Budget Experiment Implementation Plan

> 구현 위치는 이미 부착된 격리 worktree다. 기존 코드·체크포인트·발표 자료는 읽기 전용으로 사용한다. 사용자 승인 범위는 Phase A 실제 실행, 조건부 Phase B, 검증 후 정상 main 게시다.

목적은 고정 단변량 F에 학습된 정보를 전달하는 B2가 행 분할 Raw/merged LoRA에 비해 추가 선택지를 만드는지 검증하는 것이다. 실행 전 규칙은 PROTOCOL.md에 고정한다.

1. 출처와 보존: 최신 main·원본 dirty checkout·모든 기존 tracked file SHA·발표안 두 버전·checkpoint/prediction/data/scaler/origin을 확인한다. 신규 캐시만 사용한다.
2. 독립 worker: worker.py는 facet마다 모델 한 번 적재, full V parity와 FP64 재채점, whole path screen/confirm, 정확한 프로세스와 메모리 범위를 기록한다. 테스트는 chunk 순서/불균등 tail/결측/잘못된 정체성/정밀도/호출 수와 시간 범위를 검증한다.
3. 실행 제어: run.py는 기존 source context와 새 결과 cache를 구분하고 GPU 자식을 한 번에 하나만 시작한다. 요청·시작·종료·실패를 보존한다. 기존 완료 receipt의 정체성이 바뀌면 재사용을 거부한다.
4. 분석: analysis.py는 원시 timed trial에서 비용을 독립 재집계한다. 비용만으로 confirm을 선택하고, 동등한 데이터·batch·장치·메모리 계층에서만 Pareto를 계산한다. CPU Ridge는 별도 표로 둔다.
5. Phase A: GPU44+CPU2 screen 실제 실행 후 등록 규칙의 confirm과 full V 검증을 완료한다. 관측된 비용으로 남은 전체 Phase B 예산을 계산하고 진행/생략을 봉인한다.
6. 조건부 Phase B: 게이트가 통과할 때 training.py에서 Jena K3/K12×두 seed, cap3 A_fit/A_stop 선택 후 fresh A60 refit을 수행한다. 기존 K6는 그대로 재사용하고 신규20 facets의 같은 비용 검증을 실시한다.
7. 마감: 영어 그림·한국어 보고·재현 명령·실행시간·누락 사유를 기록한다. CPU 테스트와 실제 receipt/CSV 재현, 보존·공개 범위 점검 후 개별 파일만 commit하고 정상 main push 및 원격 blob 일치를 검증한다.

단계별 실제 완료 상태는 STATUS.json과 RUN_MANIFEST.csv에 기록한다. 이전 보고서를 새 결과로 대체하지 않는다.
