# 승인된 시간 앞단 실험: 실제 실행 계약

사용자가 2026-10-15에 INPUT_PLAN.md 전체와 R1–R5 실제 실행을 승인했다. 이 문서는 승인을 다시 요구하는 계획서가 아니라 실행 조건의 기록이다. 기존 캠페인은 재개하지 않는다. 기존 과학 결과를 새 결과로 재사용하지 않고, 해시가 일치하는 안전 FIT 자산과 로컬 frozen Bolt만 사용한다. 보호 TEST, 원래 TUNE의 학습 전환, 기존 작업 종료, 설치·새 모델 다운로드, 자동 commit/push는 범위 밖이다.

## 결과를 보기 전에 고정한 조건

Beijing(C11,K2), Jena(C21,K5)가 주 자료다. Robin(C17,K4)은 짧은 기간 스트레스 자료다. 첫 fold는 기존 허용 FIT 내부 60/20/20, 두 번째는 50/20/30이다. A는 모든 합법 origin을 학습 pool로 둔다. B/C는 48 간격 및 마지막 합법 origin이다. 마지막 두 창은 겹칠 수 있으므로 독립 표본으로 세지 않는다. A/B, B/C의 미래 target 경계는 분리한다. 과거 입력의 겹침은 허용한다.

스케일·PCA·초기값은 각 fold A에서 새로 만든다. Jena/Robin은 옛 전체 TRAIN 정규화 값을 FP32 역변환한 계보가 남는다. NPZ member 전체 압축 해제 때문에 옛 TUNE/VAL이 물리적으로 역직렬화될 수 있으나 반환값·새 통계·학습에는 기존 FIT만 들어간다. C는 이번 run의 선정·학습에서 제외하며, 과거에도 미노출이었던 독립 holdout이라고 부르지 않는다.

L512,H48, 원 관측 mask, A-only 표준화 채널 macro MSE, seed92601/92602, batch4, AdamW .001, clipping1을 사용한다. P는 hidden32, 잔차1블록, 내부 해상도1/2/4, kernel25이다. r=C는 항등, 작은r는 A-only PCA로 초기화한다. D는 lag0/1/2/4/8/16/32의 동일 origin-reference 선형 클래스이며 초기 lag0에 E의 의사역행렬, 나머지0을 둔다. 현재 원 C 값을 anchor로 보존한다. F의 가중치는 동결하고 입력 역전파는 유지한다. 기존 constant-window sqrt의 0-variance 미분만 유한한 0으로 정의하며 양의 분산의 수식과 순방향은 보존한다.

첫600 update, CP0/100/200/400/600이다. 동일 dataset/r/seed의 5개 방법 중 마지막 CP가 최저이거나 최근3개 CP의 B MSE가 엄격히 감소하면 같은 묶음을1200, 필요하면2400으로 연장한다. 2400에도 이 조건이면 미수렴/미해결로 표시한다. 확증 통계나 충분 수렴을 보장하지 않는다. 각 방법은 같은2400-update 스케줄의 prefix를 사용한다. 시간 전체에 퍼진 원점 pool에서 계층화 추출하며 실제 고유 origin·고유 target 시간·관측수·반복을 기록한다.

## 실제 실행 목록

1. R1: A0–A13의 원 미래48 점수. SPRINT는 metadata period(Beijing/Robin24,Jena144) 하나를 고정한다. 공식 T5 wrapper를 Bolt로 바꾼 port이며, 유효 L504/432에 맞는 Raw도 추가한다. 긴 L 비교는 동일 origin에서 Raw512를 재계산한다. 부족한 Robin L은 붙여 만들지 않는다.
2. R2: 두 주 자료 × C/K × B0–B4 ×2seed=40개의 실제 학습. B3/B4는 실제 사용하는 파라미터 수가 같으므로 추가 가짜 capacity padding은 없다. 각 실패는 해당 경로만 보존하고 다른 가족은 계속한다.
3. R3: 같은 원 C past-current 입력, hidden32, zero-output G를 Raw/B0K/B2K/B4K/current-only에 맞춘다. 앞단 단독 실패와 무관하게 두 가족을 실행한다. P/E/F/D는 고정하고 G만 학습한다. 무보정 CP0도 B 선택에 포함한다.
4. R4: B에서 고른 B2/B4의 E/D/P를 고정하고 저비용 latent linear head를 actual-Y 선형 ridge로 맞춘다. 직접 작은 TSMixer/TimeMixer, 현재값 anchor의 원 C 단변량 block ridge, 기존 q/v rank8 LoRA(alpha16,dropout0) 한 구성을 새 계약에서 비교한다. 새 모델 다운로드는 하지 않는다.
5. R5: 선정 모델·기준선의 두 번째 cutoff 전체 재적합, B4 분해/가지 제거 matched 재학습, horizon1–12/13–24/25–48, 채널·시간별 원수치, A 기준 과거 패턴 그룹, 학습곡선·gradient·rank·variance와 실패 사례를 보고한다. C 점수 이전에 모든 B 선정과 실제 전체 비용을 잠근다.

R4 direct/swap/LoRA는 최대16fit, G는 최대20fit이다. R2 조건부 연장과 문서에 명시한 ablation·중간r·second-fold refit은 독립적인 새 방법 탐색으로 늘리지 않고 지정 비교만 수행한다. 지속 clipping/발산은 사전 보조 lr.0001 한 번의 같은 비교를 허용한다. 환경·목표·자료 범위를 실질적으로 바꿀 필요가 생기면 해당 경로에 이유를 기록한다.

## 비용과 산출물

하나의 GPU queue에서 학습 후 정밀 비용을 잰다. 별도 가격·외부 서비스 비용은 없다. 실제 20-update probe는 Beijing에서 .027–.035초/update였으며, 초기 예상은 전체2–4시간이다. 조건부 연장·실패 수선에 따라 달라진다. 실장치 RTX4070 12GiB, 사전 여유 약10GiB, E: 여유 약368GiB다. 다른 Python 작업은 종료하지 않는다. 자기 프로세스 자원과 외부 부하는 기록한다.

비용은 B1/B8, warmup5/repeat10의 전체 경로와 stage를 같은 trial에서 측정한다. 원 physical CPU 입력→채움/표준화→P/E→실제 F→D/G→원단위 CPU 반환을 포함한다. Raw 채널 microbatch 정확도도 검사한다. F는 하나만 상주하고 활성 후보만 올린다. 독점 peak는 별도 프로세스에서 경로별 실행한다. resident weights와 peak allocated/reserved 및 차이를 구별하며 그 차이를 순수 activation이라고 단정하지 않는다.

DATA_CONTRACT/RUN_INDEX/SCORES/COST/DECISION_LOG/REPORT_KO/PROFESSOR_BRIEF와6종 실제 수치 그림을 한 campaign 안에 묶는다. 모든 새 시도·무효·미실행을 보존한다. 수치 검산은 저장 예측·mask·선정·좌표의 재계산 범위와 학습/타이머 독립 재실행 여부를 구별한다. C를 보고 재튜닝하지 않는다.
