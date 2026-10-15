# 고정 causal lag·디코더 학습 목적 대조의 전체 비용 사전 계약

상태: 비용 SOURCE/실제 타이머 실행 전 계획이다. [과학 계획](PLAN_CAUSAL_LAG_OBJECTIVE_KO_v1.md)의 실제 결과와 독립 PASS를 정확한 SHA로 결속한 뒤 실행한다. 원 결과·실패·다른 시행의 타이머는 보존한다.

## 측정 행렬

자료는 Robin·Beijing Aotizhongxin, batch는1·8이다. 각 자료에서 원 C-channel RAW_BOLT와 PCA/FIT_TARGET/RAW_SUBSET × L0/L7 × RECON/FORECAST의12경로, 합계13경로를 같은 시행으로 측정한다. 새 native나 전역 D 적합은 이 비용 행렬에 없다.

```text
조건 수       자료2 × batch2 × 경로13 = 52
warmup        경로·batch마다 2회
반복          13회: seeded permutation의 13 circular rotations
실제 표본     52 × 13 = 676
순서 seed     20261015
```

각 경로가13개 위치를 한 번씩 쓰도록 완전 순환한다. 같은 첫 B개 개발 origin을 반복하므로 676개의 독립 데이터가 아니다. 비정상적으로 느린 값이나 유리하지 않은 값도 제거하지 않는다. 부분 실패는13개 선언 슬롯을 유지한 채 해당 경로만 실패로 기록하고 가능한 다른 경로는 계속한다.

## 입력·실제 실행·타이머 경계

원 causal physical bridge의 원 경로·파일 SHA·member shape/dtype/SHA·origin·관측 mask·FIT mean/std를 유지한다. 원 physical alias에 member manifest가 없다면 같은 원 파일/SHA에 결속된 canonical manifest를 사용하며 loader·공차를 완화하지 않는다. 모든 새 E/D는 과학 결과의 잠금된 계수만 읽는다. FIT/TUNE/DEV로 다시 적합하거나 cost/DEV에 따라 D를 다시 고르지 않는다.

타이머 안은 physical512 창과 관측 mask에서 causal fill/FIT normalize → 실제 NumPy E 또는 raw subset slice → 실제 동결 Bolt의 K-stream 미래48 → 새 lagged delta 특징/고정 D32/current-C anchor → 원단위 FP32 CPU materialization과 최종 CUDA sync까지다. RAW_BOLT도 동일 physical 전처리와 실제 full-C F·물리 반환을 포함한다. 캐시 예측 반환을 실제 F latency로 세지 않는다.

고정 E/D/F 준비·학습·metadata/file/state SHA·배열 binding hash·source 검증·guard·warmup·저장·memory snapshot은 타이머 밖이다. origin별 D 적합은 없다. pure 함수의 실제 배포 계산/검증이 callback 안에 있으면 그 구현 비용도 포함한다. 별도의 stage 비용을 전체 latency로 바꾸지 않는다. 전처리/공차는 기존 canonical 계약을 그대로 재사용한다.

선정된 상태/입력과 과학 캐시의 수치 precheck를 먼저 수행한다. 각 timed normalized/physical 출력은 실제 과학 prediction과 별도 B1/B8 reference 모두에 대해 정규화 atol1e-5/rtol2e-6, 물리 atol2e-3/rtol2e-5를 적용한다. precheck나 timer parity 실패를 성능 결과로 받지 않으며 허용 오차를 사후 확대하지 않는다. ns start/end, 실제 시간·출력·순서·origin·양쪽 parity·shared RSS/VRAM을 표본별 보존한다. 같은 F의 모델 binary/config/dual 상태 signature와 E/D 계수는 전후 확인한다.

## 보고와 판단

전체 batch ms와 ms/window를 구분한다. 같은 반복끼리의 paired Raw ratio 중앙값과 ratio-of-Raw-medians를 모두 제공하고 혼동하지 않는다. 원 동시 기준은 각 자료·12경로의 TUNE full-observed normalized macro MSE/MAE가 각각 Raw×1.01 이하이고 B8 ratio-of-medians≤.90인지 확인한다(자료·경로24개). 이 기준은 새 D나 alpha/lag/목적 선택 규칙이 아니다. 이미 지정한 모든 경로를 보고하며 DEV 최저 경로를 채택하지 않는다.

같은 process에 모델과 여러 E/D가 resident이므로 RSS/VRAM은 방법별 독점 할당량이 아니다. GPU kernel 시간이나 장비 독점·조용한 환경을 주장하지 않는다. 새 고정 D와 이전 origin-fit D 비용은 경로가 다르다. 다른 cohort의 절대 Raw 시간을 분모로 가져오거나 가장 빠른 run만 고르지 않는다. F inference, D 적합·전체 학습, 저장량, 논리적 입력 정보량을 나누어 기록한다.

독립 감사는 saved ns/표본/상태/예측/원 단위·순환순서·모든 원 경로를 검산하며 F/GPU timer를 몰래 재실행하지 않는다. 자료 부재·구현·수치·자산 문제는 해당 경로의 상태로 남기고 독립 가능 경로를 계속한다. 최신 계속 실행 지시를 따르되 TEST·설치·큰 모델 학습·기존 작업 종료/재개·자동 Git 게시를 포함하지 않는다.
