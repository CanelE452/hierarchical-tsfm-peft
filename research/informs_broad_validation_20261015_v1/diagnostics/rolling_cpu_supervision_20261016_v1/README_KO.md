# 실제 CPU 두 slot 실행 감독 코드

진행 중인 새15 fit·22 V 출력 큐에 실제 사용한 [rolling_fit_queue.py](rolling_fit_queue.py), 종료·identity 검사에 사용한 [cpu_sequence.py](cpu_sequence.py), [합성 검사](test_rolling_fit_queue.py)를 원byte 그대로 보존했다. [SOURCE_BINDING.json](SOURCE_BINDING.json)은 실제 source/CONFIG SHA와 복사 범위를 기록한다. 원 single-worker guard는 [공개 원본](../../postT_V_only_cpu_lag_20261016_v1/cpu_group_supervisor.py)이다.

모든 과학 단계는 single-child guard로 실행하고 최대2 slot·각 CPU2 threads/샘플링 RSS3GiB를 유지한다. 각 fit의 종료를 확인한 뒤 자기 V·독립 재채점·새 CPU replay까지 같은 slot에서 처리한다. 닫힌 개별 실패는 원본을 보존하고 다른 독립 조건을 계속한다. 종료 미확인·source/plan 변경·전체 자원 상한은 새 입장을 중단한다. fixed nested3은 optimizer0의 독립 보조 비교이며, 공통 PCA basis 생성 경합을 막기 위해 V 생성 단계 전체를 직렬화한다. 학습 중 비용 벤치마크를 수행하는 큐가 아니다.

이 source 보존은 다른 컴퓨터에서 현재 private CONFIG를 그대로 실행하라는 지시가 아니다. CONFIG는 현재 자산·경로·prior identity·봉인된 probe·deadline에 결박되어 공개하지 않았다. 새로 실행하려면 실제 자산과 공개 PLAN/runner의 조건을 확인하고 충돌 없는 새 ID·경로·resource 계약과 실제 probe를 구성해야 한다. 기본 CLI는 metadata-only이고 실제 실행은 명시 CONFIG SHA 및 ROOT trigger를 요구한다. 이미 실행 중인 ID를 다시 시작하거나 기존 실험을 재개하는 용도로 사용하지 않는다.

이 캡처에서 fixed nested3의 V/재채점/replay는 완료됐지만 새15 fit·그 뒤22 V·신규 비용 전체의 완료는 아니다. source 복사·합성 PASS는 학습 성공·수렴·독립 일반화나 원 연구 완료의 증거가 아니다.
