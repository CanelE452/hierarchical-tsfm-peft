# K 전환 측정 경계 수정 검증

CPU 합성 실패 주입으로 deadline 이후 예측 시작과 후처리 실패 시 기존 trial 손실을 재현한 뒤 수정했다. 기존36PASS/기존 코드3개는 보존했다. 마지막 예측 후 자원 초과는 수치 수행량을 지우지 않고 resource-valid ledger와 최종 protocol 미완료로 분리한다. 최종 close/완료 직전에도 owner를 검사하고 초과 뒤 새 strategy를 시작하지 않는다. memory/state/summary 실패가 원 numerical rows와 ledger를 덮지 않는다. 개별 trial JSON·CSV 값·ledger도 대조한다.

fixture는 비영 미래 F와 채널별 비선형 과거·명시적 E/D로 K별 예측이 달라지도록 변경했다. 올바른 stream 수만 맞춘 wrong E column/flat D prefix가 canonical mismatch로 탐지된다. 전체 실제 GPU·원자료·보호T·실제 checkpoint/예측 재실행은 하지 않았으며 새 코드의 실측은 ROOT가 별도로 결정한다.
