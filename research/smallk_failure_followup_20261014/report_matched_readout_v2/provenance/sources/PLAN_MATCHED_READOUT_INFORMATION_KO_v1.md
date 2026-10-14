# 다음 승인용: 원 C 입력과 K 표현·시간 요약을 같은 readout에서 비교

상태: SOURCE_PLAN_ONLY / 미실행. 현재 단계의 [결과 해석](STAGE_SEPARATION_INTERPRETATION_KO_v1.md)을 바탕으로 정한 다음 범위다. 기존 승인·manifest·결과를 연장하거나 덮어쓰지 않는다.

목적은 고정 E가 정보를 잃는 것인지, 33K block mean 요약이 정보를 잃는 것인지, 아니면 이 readout의 예측 한계인지를 좁히는 것이다. 원 E/F/D 재학습·대조학습·동적 그래프 추가는 포함하지 않는다.

Robin C17/K4 및 Beijing C11/K2 × 원 seeds92601/92602, 원 FIT256/TUNE64/VAL30 또는 EVAL146, L512/H48, 원 통계·origin·actual-Y mask를 유지한다. 배포 입력은 과거뿐이다. 미래 oracle teacher는 새 학습에 쓰지 않는다.

기존 N2 E33K 결과를 SHA 검증 후 재사용한다. 신규는 각 원조건에서 RAW33C와 E512K 두 경로, 총8 fits × 고정7 alpha의 최대56 후보다. RAW33C는 원 normalized 과거 C의 32개16-point mean과 마지막 값을 펼친다. E512K는 기존 원 E의 모든512 latent 시점을 그대로 펼친다. 각 경로는 동일 actual-Y innovation 목표·current-C anchor·unpenalized intercept·FIT-only feature scaling·channel-macro ridge 손실·actual-Y TUNE MSE 선택을 사용한다. alpha는 0, 1e-4, 1e-3, 1e-2, .1, 1, 10, exact tie는 이른 후보다.

입력 차원·유효 rank·계수 수가 다르므로 성능 차이를 순수 encoder 효과로 확정하지 않는다. RAW33C 대비 E33K는 현재 표현+축약 경로의 유용성, E512K 대비 E33K는 더 많은 과거 latent와 readout capacity를 유지한 효과를 보는 조건부 비교다. 후보 전부의 FIT/TUNE와 선택된 노출 평가만 저장한다. 새 결과를 보고 알파·feature·mask를 바꾸지 않는다. 기존 raw controls·RAW·MAIN도 재사용하며 없는 비용을 0으로 쓰지 않는다.

원 CP·기존 cached latent·원 input bundle의 실제 존재와 SHA부터 확인한다. solver는 동일 목적을 유지하며 고차원일 때 dual/SVD 구현의 등가성을 synthetic fixture로 먼저 확인한다. 미관측 output은 innovation0으로 표시한다. source/coeff/input stats/count/rank/정규화·물리 오차/학습 process wall/검산을 새 경로에 남긴다. 보호 TEST, 중단 학습 재개, F 재학습, 자동 commit/push는 하지 않는다.

제안 자원은 추가 CPU process15분·wall30분·저장1GiB, GPU0이다. 고차원 solver와 자료 검산이 이 범위를 넘으면 해당 실제 상태를 보존하고 종료한다. 전체 완료나 성능 개선을 보장하지 않는다. 새 human 승인과 고정 실행 manifest 전에는 실제 자산 읽기·solver·fixture 실행을 시작하지 않는다.
