# 고정 실행 계획

사용자가 채택한 [실행 계약](REQUEST.txt)을 실제 로컬 파일에 연결합니다. 시작 main/원격 SHA는 `317bd62e41c6eeaf670c9c75f0e5d3f035bf1350`입니다. 기존 exact-contract Mini/Tiny 완료 산출물은 research·untracked·cache 검색에서 찾지 못했고 관련 실행 프로세스는 관찰되지 않았습니다. Windows GUI GPU 사용은 존재하며 비용 drift는 sentinel과 원블록에 보존합니다.

## 고정 arm과 모델

- BOLT_SMALL_LEVEL: 기존 LEVEL 두 선택 seed, fixed TRAIN PCA C→K와 G 그대로. Robin v6, Jena v7, Peacock v12. 이전 TEST 예측 재사용.
- BOLT_SMALL_F0: 기존 원채널 small zero-shot 한 개. 이전 TEST 예측 재사용.
- BOLT_MINI_F0: `amazon/chronos-bolt-mini`, revision `251268337516a88e253628c43e1d26ec577b376b`.
- BOLT_TINY_F0: `amazon/chronos-bolt-tiny`, revision `a0e552de83495b5c28c14c71c374f3e33280b340`.
- 기존 Small revision은 `772f3d25d38aec6d914c8949dab4462e2d46f5d8`입니다. 각 공식 config Git blob 및 가중치 LFS SHA와 실제 파일 SHA를 대조하고 실제 로드 params/bytes/buffers를 기록합니다.

## 실행과 판정 게이트

1. 기존 data/checkpoint/prediction/source의 hash·형상·원점·mask를 바인딩합니다. 데이터 채널·TRAIN 통계·결측 대체·경계는 바꾸지 않습니다.
2. 각 Mini/Tiny×세 자료의 첫4 VAL 원점에서 shape·finite·q0.5·채널 독립·원점 격리·단독/B4·순서복원·미래 차단·frozen state·재호출을 실제 forward로 확인합니다. 일반화 Small wrapper와 세 자료 기존 F0 wrapper도 대조합니다. replay atol1e-6/rtol0, batch atol1e-5/rtol1e-4를 그대로 사용합니다.
3. 구현·모델·자료·재사용·preflight를 TEST 전에 봉인합니다. Mini/Tiny 두 모델×세 자료의6 inference units만 canonical A→B origin 순서로 B4 실행합니다. 완성 receipt가 있으면 재실행하지 않습니다. 중단분·실패는 보존합니다.
4. 기존 error-term/channel-macro scoring과 A/B-stratified paired moving-block bootstrap을 재사용합니다. block7/7/42, draws2000, seed9262026, salts0/0/1입니다. LEVEL 두 seed의 손실을 평균하며 zero-shot seed를 복제하지 않습니다. A/B/combined MSE·MAE·bias·채널·원점과6개 대응 비교를 남깁니다.
5. 비용은 새 캠페인에서 네 arm 모두 측정합니다. LEVEL 두 seed와 F0/Mini/Tiny 각1개를 포함한5instances×B1/B4×3blocks×3datasets=90행, F0 B1 pre/post sentinel18행으로 정상 완료 예정108행입니다. 첫24 chronological VAL origins, warmup10, 전체24-origin passes10, blocks3와 순서회전을 고정합니다. FP32·TF32 off·CPU4·RTX4070, compilation/quantization 없음. 표준화 CPU 입력→transfer/reshape→전체 online 경로→contiguous CPU H48C 출력까지 측정하고 load/hash/score/log는 제외합니다.
6. 공정성 표, 결과/상대차이/조건부 구간, B1/B4·allocated/reserved·resident·params/bytes·drift/원블록과 Pareto 판정을 작성합니다. 모든 비용축이 같지 않으면 단일 지배로 쓰지 않습니다. 노출된 후속 비교·자체검산·사전학습 중복 미확인의 한계를 유지합니다.
7. stored scoring 재검산, preflight/seal/model/data/old-prediction binding, cost input hash/모든 required arms·3blocks, 원본 보호, 설정 불변·실패 보존, Git related-file allowlist를 확인합니다. final_checks를 읽고 관련 코드·작은 결과·보고만 main에 ordinary commit/push합니다.

## 자원과 중단

현재 사용자의 고정 비교 요청 안에서 운영상 hard cap을 GPU job7200초, CPU job3600초, 새 모델 파일512MiB, 새 폴더2GiB로 설정합니다. 이는 이전 회차의 잔여 예산을 합친 승인이 아닙니다. 더 큰 자원이 필요하면 STOP_RESOURCE로 멈추며 임의 확대하지 않습니다. 원장 시간은 계측 작업의 wall time이고 대화·전체 개발 시간이 아닙니다. 다운로드는 새로 materialize한 파일 bytes를 보수적으로 계상하고 network wire 계측이라고 주장하지 않습니다.

정의·parity·모델/자료/원점/미래 차단·state·이전 예측 binding 실패는 STOP_DEBUG입니다. 기술 path/API/serialization 수리는 계약을 유지할 때만 기록 후 허용합니다. OOM은 그 행의 결과로 남기고 B4를 임의 B2로 바꾸지 않습니다. 불리한 TEST 때문에 설정·revision·L/H/K/rank/모델을 바꾸지 않습니다.
