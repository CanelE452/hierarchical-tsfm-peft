# 원 채널 subset·혼합 공간의 고정 과거 디코더 전체 비용

상태: 실제 비용 실행 전 계획이다. 원 채널 subset 비교의 실제 선택·추론·학습·독립 검산을 먼저 마친 뒤 새 비용 source를 고정하고 실행한다. 계획을 비용 측정 완료로 세지 않는다.

## 목적과 비교 조건

[판단] `PLAN_RAW_CHANNEL_SUBSET_KO_v1.md`의 정확도 비교와 함께, origin마다 디코더를 적합하지 않는 고정 FIT 디코더 경로가 원 채널 Bolt보다 실제로 저렴한지 확인한다. 기존 shrinkage의 비용이나 이전 시행의 Raw 시간을 재사용해 speedup을 계산하지 않는다. 같은 작은 K·기존 원 데이터/정규화/관측 mask/과거512·미래48·현재 C anchor·frozen Bolt 조건을 유지한다.

자료2개(Robin C17/K4, Beijing Aotizhongxin C11/K2), batch1/8, 경로10개를 같은 process에서 대조한다. 경로는 Raw Bolt와 PCA/FIT-target/subset 각각의 Bolt 및 새 native seed92601/92602다. 총40셀이다. 각 셀 warmup2회, 실제10회로400 timing samples를 계획한다. seed20261015로 정한10경로의 기준 permutation을 각 반복에서 순환시켜10회의 모든 순서 위치를 균등하게 사용한다. 이는 같은 첫 B개 개발 origins를 반복하는 기술적 측정이지 독립 데이터400개가 아니다. 모든 원 sample과 outlier를 저장한다.

본 비교의 native·subset 선정·global D·checkpoint는 승인된 원 정확도 실행의 TUNE 잠금 이후 고정한다. 비용/DEV를 보고 새 K·채널·seed·checkpoint·E/D를 고르지 않는다. 같은 decoder fitting rule이지 세 E의 수치·감독 정보가 동일하다는 뜻이 아니다. FIT-target E 자체의 기존 FIT 실제 미래 감독과 subset의 FIT 과거-only 선정 차이를 유지한다.

## 실제 경계와 확인

타이머는 physical window의 causal fill 전부터 FIT mean/std 정규화, 실제 해당 backend E, 실제 frozen F 또는 선택 head forward, 고정 FIT D32를 통한 `currentC + (futureZ-lastZ) @ D32`, 원 단위로 복원하여 CPU 반환값을 materialize하고 CUDA synchronize하는 단계까지 포함한다. Subset은 실제 원 채널 slice를 사용하며 K-channel F를 호출한다. Full-C F 결과를 slice하여 K-channel inference를 대체하지 않는다. native는 실제 Torch CPU2 실행이며 batch1/8의 수치 경계와 canonical checkpoint batch32의 차이를 precheck에 남긴다.

고정 E/D/head를 미리 로드하는 것은 배포 모델 준비다. 본 경로에는 per-origin D 적합이 없으므로 매 callback 적합 비용도 없다. FIT의 E/subset/global D 적합·native 학습 비용은 과학 실행 receipt 및 training metadata로 별도 보고한다. 모델/state/file hash, setup·원 예측 참조·warmup·저장·guard·memory snapshot은 타이머 밖이다. 물리 입력과 원 선택 예측의 정확한 canonical array manifest·path/SHA를 검증하며 기존 alias의 I/O 검증을 생략하지 않는다.

실행 전 같은 first8 origins의 물리 전처리·E/D backend 계약·선정 예측과의 오차·F 파일 hash를 확인한다. Shape/dtype/mask/scaler/origin이 다르면 해당 경로만 실패로 보존한다. 허용오차는 기존 실행의 실제 precheck 계약을 source 고정 시 명시하며 실패 후 수치를 좋게 보이도록 바꾸지 않는다. 시작/종료 시 F와 모든 원 고정 E/D/head 및 원 과학 결과·선정 잠금의 SHA가 같아야 한다.

각 셀에 whole seconds의 원 ns start/end, 중앙값, ms/window, 같은 시행·batch의 Raw 중앙값 대비 비율, 같은 반복끼리 짝지은 Raw 비율의 중앙값을 각각 저장한다. 두 비율을 혼용하지 않는다. RSS/VRAM은 모든 비교 모델이 resident인 공유 process의 계측값이고 방법별 독점 메모리나 실제 대역폭의 감소가 아니다. 환경 전체를 독점한다고 주장하지 않는다. 이전 Raw latency 변동은 보존하고 별도 cohort의 절대 시간을 결합하지 않는다.

독립 auditor가 timer ns 연산·10회 순서 균형·400 표본·실제 callback output·정확한 물리 입력 binding·40셀 완전성·모든 원 체크섬을 검산한다. 미완료 경로는 실제 상태/오류/재개 경로를 남긴다. 보고서는 원 주지표 TUNE MSE/MAE 각각 Raw1.01배 이내 및 같은 시행 B8 ratio-of-medians .90 이하라는 기존 동시 기준도 검산하되, 이 기준 통과 자체를 독립 일반화로 바꾸지 않는다.

## 권한과 보존

[판단] 최신 직접 지시 “상한내수행하지 말고 쭉해줘”에 따라 기존 시간·누적 자원 상한 없이 같은 연구 목적의 비교를 지속한다. 이번 계획에 대한 별도의 exact human reply가 있었다고 주장하지 않는다. 실제 process 사용량은 계속 계측하고 원 승인/실패/자원 영수증을 보존한다. TEST·설치·큰 backbone 학습·기존 실행 중지/재개·자동 commit/push·논문/PPT 작성은 포함하지 않는다.
