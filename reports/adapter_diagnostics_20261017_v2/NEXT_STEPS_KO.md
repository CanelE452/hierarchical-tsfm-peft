# 결과 확인 후의 후속안

[확인] B01 실제 관측: seed=92601 ΔMSE=2.2751504656630277%, ΔMAE=2.081045706128617% (EXACT_MATCHED); seed=92602 ΔMSE=0.7074761791166884%, ΔMAE=0.9324235205081115% (EXACT_MATCHED). 모든 완료 seed에서 B2 probe의 두 오차가 B0보다 높다. λ1 고정 full-history reader의 조건부 접근성 결과이며 표현 정보 전체의 부재나 우위로 확대하지 않는다.

[확인] B03 실제 관측: seed=92601 ΔMSE=1.9632759547043914%, ΔMAE=-0.20884904129536955% (EXACT_MATCHED); seed=92602 ΔMSE=0.42188567468289156%, ΔMAE=-0.653045780727166% (EXACT_MATCHED). 모든 완료 seed에서 Dh는 shared보다 MSE가 높고 MAE가 낮다. Dh와 shared의 오차 방향·용량·선택 regularization을 함께 읽고 단일 원인으로 단정하지 않는다.

C02 초기화: 등록된 2개 평가가 rows와 완료 영수증에 있다. 같은 pair/trio의 재실행을 자동 추가하지 않으며, 남은 질문은 용량·선택 학습량·반복 V의 한계를 분리할 수 있는지다.

D02 두 번째 단변량 backbone: 등록된 2개 평가가 rows와 완료 영수증에 있다. 같은 pair/trio의 재실행을 자동 추가하지 않으며, 남은 질문은 용량·선택 학습량·반복 V의 한계를 분리할 수 있는지다.

D03 native multivariate 대안: 등록된 2개 평가가 rows와 완료 영수증에 있다. 같은 pair/trio의 재실행을 자동 추가하지 않으며, 남은 질문은 용량·선택 학습량·반복 V의 한계를 분리할 수 있는지다.

C01 원 CPU exact continuation은 남은 장치 재현 질문이다. [확인] CUDA numerical branch pair는 이미 완료했으며 역사적 CPU bitwise continuation과 구분한다. 기존 CPU 단일 next-update restore 검사와 원 CPU search/refit 전체 완료는 별개다. 최소 추가 단위는 원 CPU 동일 device B0/B2 pair이며, [추정] profiles/C00_pems_pair_cpu.json (SHA256=f35fee71f53902fa4395b3282ae912fdd6b3576abbea9105a8009c59563abfce)의 conservative_complete_pair_seconds=76306.50345109217초, 약 21.20시간. 실제 disposable update profile의 보수적 외삽이며 전체 완주 실측이 아니다. 이는 현재 6시간 전체 캠페인 예산 밖이므로 별도 장시간 예산 승인 전 실행 대상으로 삼지 않는다.

이번 고정 계획을 자동 확대하지 않는다. 남은 승인 큐의 status/reason/dependency와 실제 잔여 예산을 먼저 확인하고, 예산 종료 뒤 추가 실행은 새 승인 예산에서 위 최소 pair/trio/cohort 단위만 검토한다. 기존 deadline/사용량을 초기화하거나 자동 대형큐를 재개하지 않는다.

D01 합성 smoke/profile 시간을 Moirai 또는 Chronos2의 실제 전체 학습·평가 시간으로 외삽하지 않는다. Moirai의 후속 비용은 해당 실제 training profile/측정 범위를 따른다. Ridge의 정확도·비용 우위가 확인된 cohort는 실용적 약점으로 유지한다. 새 독립 test/generalization 평가의 자료 확보·학습·전체 비용은 UNKNOWN이며, 개발 V의 개선을 독립 일반화로 바꾸어 주장하지 않는다.
