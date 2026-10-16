# INFORMS 전체 실행 계약과 순서

이 문서는 **진행 중 실행 계획**이다. 완료 보고서나 결과의 추가 가치 판정이 아니다. 실행 사실은 개별 receipt, 저장 체크포인트 해시, 예측 manifest, 독립 검산 결과로 확인한다.

사용자가 제공한 [START_HERE](plan/START_HERE.txt), 전체 계획, [계약 템플릿](plan/CAMPAIGN_CONTRACT.yaml)을 전부 확인했다. 원문 계획 MD의 SHA256은 `34d242626f163a8c510d4f6750170bdf9a34af87a610b6aee9df7781fb708569`이다. [공개용 전체 계획](plan/MASTER_PLAN_PUBLIC.md)은 개인 상담의 시간대·요약 한 문단만 제외했으며 실험·권한·예산 계약은 그대로 보존한다. [사본 계보](plan/PUBLIC_PLAN_PROVENANCE.json)를 남겼다. 실제 자산 조사 후 [CONFIG_LOCK](CONFIG_LOCK.json), [DATA_CONTRACT](DATA_CONTRACT.json), [초기 등록표](EXPERIMENT_REGISTRY_INITIAL.csv)를 과학 학습 전에 고정했다. 현재 실행 상태는 [등록표](EXPERIMENT_REGISTRY.csv)에서 별도로 관리한다.

## 목적과 주장 한계

고정된 단변량 Chronos-Bolt를 여러 변수에 사용할 때, 어떤 자료·K·학습 노출에서 외부 표현과 예측 결합에 정확도–비용의 추가 가치가 있는지 확인한다. B0, B2, B4, 지연 압축, 조건부 보정, 다중 K를 서로 다른 질문으로 다룬다. 한 방법의 음성 결과로 다른 원리의 최소 실험을 생략하지 않는다. 검사 통과는 성능 개선이나 독립 일반화의 증거가 아니다.

기존 R1–R5의 main 기준점은 `c7a06e522e1cd8e100a3e2291a7f693dd3f7125f`이다. 기존 결과는 보존하고 [재사용 판정](REUSE_MATRIX.csv)에 따라 역사적 개발 근거로만 쓴다. 새 시간 분할·scaler·calibration 조건이 달라 새 캠페인의 성능 값으로 대체하지 않는다.

## 자료와 보호 경계

Beijing 11채널, Jena 21채널, Electricity 표준 hourly 321채널, PeMS08 flow 170채널이 실제 로컬 자산으로 확인됐다. Robin은 짧은 자료의 보조 스트레스 실험이다. Jena는 10분, Beijing/Electricity는 1시간, PeMS는 5분 간격이며 같은 512행의 물리적 시간 길이가 다르다. M5 3패널은 공식 경로의 인증 부재로 차단됐으며 다른 출처로 몰래 대체하지 않는다. 자세한 획득·보호·라이선스 상태는 [자료 감사](DATA_ASSET_REVIEW_KO.md)에 있다.

Electricity의 기존 보호 시작 시각은 새로 받은 321채널 자료에도 적용했다. 원본의 보호 시간대 꼬리는 숫자 배열로 변환하지 않았다. 공개 원자료, 개인 상담 내용, 연락처, 모델 백본 가중치는 게시하지 않는다. 역사적으로 본 자료의 새 분할을 연구자에게 전혀 새로운 일반화 검증으로 부르지 않는다.

## 학습과 평가

허용 구간 안에서 A_fit/A_stop/M_cal/V_select/T_final을 50/10/10/10/20%로 나눈다. A50 학습·A_stop checkpoint 선정 후 **새 초기값과 A60 통계로 다시 적합**한다. 결합 부모는 M 이전에 고정하고, M dense stride1 예측으로 alpha/H를 학습하며 V strideH로 선택한다. 별도 A+M 재적합 G/LoRA 대조군은 정규화 척도만 A60으로 유지한다. T는 전체 선택 잠금 뒤에만 평가한다. 각 예측 target은 구간 안에 완전히 들어와야 하며 마지막 중복 origin을 덧붙이지 않는다.

주지표는 관측 target만 사용한 원채널별 정규화 MSE의 채널 macro 평균이다. MAE, 물리 단위 보조오차, 관측 수, 채널별/예측 거리별 통계를 함께 남긴다. 미래 target이나 전체 자료로 scaler/PCA를 적합하지 않는다. 결측 입력은 각 과거 창 안에서만 forward-fill하고 첫 관측 전은 A평균으로 채운다.

실행은 0.25 epoch의 폭넓은 선별 → 1/3 epoch → 같은 조건의 추가 학습 순서다. `PARTIAL_SCREEN025`는 계획된 3 epoch 완료가 아니다. 마지막 checkpoint가 계속 좋아지면 학습 미해결을 별도로 기록한다. 학습 경계의 큰 변화는 새 ID로 남기며 이전 결과를 덮지 않는다. 최종 seed와 두 번째 시간 절단의 범위도 등록표/재개 큐에 남긴다.

## 실제 실행 순서

1. E0의 기존 12개 미해결 checkpoint와 Adam 상태를 확인하고, 대응 대조군을 포함한 후속 학습을 새 경로에 수행한다. 과거 RNG가 없는 재개는 `RESTARTED_NOT_EXACT_RESUME`로 표시한다.
2. E1의 접근 가능한 자료에서 Raw/G/LoRA/PCA/B0/B2, B4, 직접 예측기, native ridge, AdaPTS 원리 포트를 실제 실행한다. quarter/half/C 및 기존 작은 K를 먼저 채우고 전체 K grid와 채널 부분집합으로 넓힌다.
3. 부모가 생긴 자료부터 E2의 alpha0/0.5/1/M-fit, B2/B0/Raw 조건부 H, 같은 용량 과거 H, 기존 잔차 방식, A+M G/LoRA를 실행한다. Raw의 M-only 채널별 상수 bias 보정도 비교한다. 이 단순 보정을 최적 Raw 보정이라고 주장하지 않는다.
4. E3는 Beijing과 PeMS08에서 causal/bidirectional 7-lag를 quarter/half/C로 실제 학습한다. 같은 D/anchor/K를 유지하고 prefix·실제 F행 수를 검사한다. 그다음 다른 자료로 확대한다.
5. E4는 Beijing과 Electricity에서 V의 사전 규칙으로 B0/B2 가족을 정하고 공유 다중 K, 최대 K 전용 PCA 초기화 모델의 사후 절단, nested PCA, 독립 K를 비교한다. 독립 K=C 항등 초기화와 공유 전체 PCA 초기화의 차이는 별도로 보고한다.
6. 충분 학습과 최소 범위를 검토한 뒤 선택 잠금, T 추론, 독립 재채점, 대표 origin 체크포인트 재추론, B1/B8 전체 파이프라인 비용, 그림·한국어 보고서, 안전한 main 게시와 원격 검증을 수행한다.

## 자원과 기록

실제 GPU 소유 프로세스 wall을 누적 12시간까지 보수적으로 계상하며 마지막 1시간은 검산·비용용으로 예약한다. 전체 wall 48시간, 새 저장 공간 30GiB, 무료 공식 다운로드 5GiB, 공개 단일 파일 20MiB·증분 250MiB 상한을 지킨다. 프로세스 lease/서명된 child handshake와 실제 장치 stream 수를 사용한다. 외부 실행 중 작업은 중지하지 않는다. 자원 종료 시 미완료 경로를 완료로 바꾸지 않고 `PARTIAL`과 정확한 재개 명령을 게시한다.

학습 업데이트, 고정 PCA의 optimizer 0회, ridge의 닫힌형 해법, 저장 예측 재채점, 실제 재추론, 비용 측정은 각각 다른 실행 사실이다. 저장 예측만 재사용한 결합 계산의 F 호출은 실제로 0이며, 배포 비용에는 필요한 부모 F를 반드시 포함한다. ridge native NumPy FP64 CPU와 GPU FP32 경로의 장치/dtype 차이도 명시한다.

공개 결과는 원점×채널×예측거리 band의 SSE/SAE/count와 코드·설정·해시·보고서·그림이다. 예측/target 전체 배열과 백본은 private cache에 남긴다. 독립 FP64 검산, 오류 주입, checkpoint subset 재추론을 각각 따로 보고하고 학습 전체의 독립 재실행이라고 표현하지 않는다.
