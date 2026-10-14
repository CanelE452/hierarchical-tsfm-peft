# 다음 범위: MAIN 예측을 보존한 과거 K 표현의 보정 유용성

상태: **SOURCE_PROPOSAL_ONLY / 미승인·미실행**. [현재 검산 결과와 판단](MATCHED_READOUT_INTERPRETATION_KO_v1.md) 및 [새 보고서](report_matched_readout_v2/REPORT_KO.md)를 근거로 작성했다. 기존 승인·시계·자료·계수·결과를 연장하거나 덮어쓰지 않는다.

목적은 현재값에서 예측을 다시 만드는 ridge가 기존 MAIN의 예측 능력을 버려 실패하는지, 고정 E의 과거 표현에 MAIN 오차를 보정할 정보가 있는지를 나누는 것이다. 원 K와 원 E/F/D를 유지하므로 원 C를 모두 TSFM에 넣는 방법으로 목표를 대체하지 않는다. 현재 C anchor는 원 MAIN에도 존재하는 추가 정보이며 strict K-only라고 부르지 않는다.

데이터는 동일 Robin C17/K4 및 Beijing Aotizhongxin C11/K2 × seeds92601/92602, L512/H48, FIT256/TUNE64, 반복 VAL30/노출 EVAL146이다. FIT 통계·원 origin·실제 Y 관측 마스크·원 CP/codec/F state·cached 과거 latent를 유지한다. 보호 TEST, 미래 oracle, 다른 자료/창/seed는 사용하지 않는다. 원 cached MAIN 예측은 과거에서 생성된 미래 예측이며 실제 미래 Y와 구분해 CP·original_prediction SHA 및 원 과거 입력 계보를 확인한다.

새 FIT 목표는 **FP32(Y − frozen MAIN forecast)**이다. inference는 **FP32 MAIN forecast + FP32 ridge correction(past features)**다. 원 E33K의 32개 16-point mean+last, E512K 전체 512시점, 그리고 원 C의 RAW33C mean+last를 비교한다. RAW33C 보정은 단순 원 자료 보정만으로 충분한지 확인하는 대조이며 원 C를 TSFM에 새로 넣지 않는다. 이 경로는 현재값 외 원 C 과거도 사용하므로 K 표현 단독이라고 해석하지 않는다. 모든 특징은 FIT-only center/scale을 사용한다. 계수는 같은 unpenalized-intercept/channel-macro ridge, lambda=alpha*Nc, 동일 mask별 thin SVD/minimum-norm 정의를 유지한다. 실제 미래 Y와 마스크는 학습·점수 계산에만 사용하고 추론 함수 인수로 받지 않는다.

각 새 경로의 후보는 **보정0 → FIT 평균 H×C 오차 보정 → alpha=0,1e-4,.001,.01,.1,1,10의 7 ridge** 순서로 총9개다. alpha0과 보정0은 다른 방법이다. FIT 평균 보정은 원 조건마다 한 번 계산하고 세 경로에 같은 값으로 제공한다. 선택은 실제 Y의 정규화 TUNE macro MSE 최솟값, exact tie는 위 순서다. 보정0 선택은 새 학습 생략이 아니라 TUNE에서 새 보정을 거절한 것이다. 평균 보정이 선택되면 past representation의 추가 정보가 입증됐다고 말하지 않는다. 각 경로의 실제 선택 종류와 평균 보정을 전부 보고하고 평가 오차를 보고 사후에 우승 방법을 선택하지 않는다.

실행 목록은 신규12 ridge route fits ×7 alpha=최대84 새 계수 후보, 보정0 대조4조건, 공유 FIT 평균 보정4조건, 경로별9개 후보를 센 총108 candidate comparisons, 모든 후보 FIT/TUNE 점수와 선택된 원 VAL/EVAL 실제 추론이다. 공유 대조를 새84 ridge 계수로 세지 않는다. 새 E/F/D·Bi-Hash·graph·큰 backbone 재학습은 하지 않는다. 기존 source/state/결과는 보호하고 fresh producer 경로에만 저장한다. 경로의 자산·검사 실패는 해당 경로에 적용하며 독립 조건을 계속한다. 자원 상한 또는 실제 활성 lease 충돌이면 다음 job을 시작하지 않는다.

먼저 zero-baseline exact parity, FP32 residual/addition, H×C/Nc·zero observation, FIT-only state, selection tie, no-future API의 synthetic 검사를 새 승인 runtime에서 수행한다. 실제 저장 배열 독립 감사는 84 ridge 후보·경로별108 비교·FIT/TUNE·선택 예측·MAIN/RAW/공유 평균보정·계수/feature SHA·origin/mask/stats·선택 계보를 확인한다. 원 F 재학습/독립 solver 최적성 증명으로 확대하지 않는다.

배포 비용은 사전에 고른 seed92601만 사용해 Robin/Beijing ×batch1/8 ×6방법 RAW-F/MAIN/평균보정/RAW33C보정/E33K보정/E512K보정의 **24 whole cells**를 측정한다. 여섯 방법이 각 위치에 한 번씩 오는 Latin 순서, warm2/repeat6, 원 normalized/physical 반환 계약, E+F+D와 CPU 보정·모든 전송을 포함한다. 같은 batch에서 예측값·관측 점수를 확인하고 저장 출력/일정/median을 독립 검산한다. 선택된 보정0/평균 보정 경로는 실제 배포에서 불필요한 feature/readout을 건너뛰며 이를 기록한다. seed92602의 배포 비용과 안정적인 장비 전반 속도 우위는 측정 범위가 아니다. 일부 방법이 실패해도 확인 가능한 독립 방법의 측정을 보존한다.

성공 해석은 원수치·각 seed·TUNE/VAL/EVAL 방향·평균 보정과 비교·실제 비용을 함께 본다. TUNE의 baseline 포함은 평가 개선을 보장하지 않는다. p132/2048 또는 p66/1024의 용량·규제 차이, FIT256의 시간 중복과 창 상관, 반복 평가의 선택 편향은 남는다. 개선이 있어도 새 독립 자료 검증 전에는 교수님께 일반화·완성된 방법론이라고 말하지 않는다. 개선이 없으면 미래 정렬·graph 전체의 불가능성으로 확대하지 않는다.

제안 상한은 **추가 CPU process15분, GPU process5분, 전체 wall40분, 저장 증가1GiB, RSS8GiB**다. CPU에는 초기 검사·fit·실제 CPU apply·감사·보고·최종 검증을 포함하고 GPU에는 원 F를 불러온 실제 whole 추론/비용을 포함한다. wall은 새 human 승인 기록 후 첫 metered process birth부터 고정한다. 모든 기존 receipt·고정 시계는 그대로 보호한다. 설치·외부 유료 자원·새 다운로드·TEST·기존 중단 학습 재개·논문/PPT·자동 commit/push는 포함하지 않는다. 이 범위에서 성공이나 전체 목표 완료를 보장하지 않는다.

이 계획과 새 pure-array prototype은 리뷰 대상이며 실제 승인 기록이나 실행 결과가 아니다. 새로운 목적/anchor와 F 비용 측정, 이미 만료된 wall을 바꾸므로 사용자 응답 후에만 새 AUTH/manifest를 작성하고 실제 자산 읽기·solver·fixture·GPU 실행을 시작한다.
