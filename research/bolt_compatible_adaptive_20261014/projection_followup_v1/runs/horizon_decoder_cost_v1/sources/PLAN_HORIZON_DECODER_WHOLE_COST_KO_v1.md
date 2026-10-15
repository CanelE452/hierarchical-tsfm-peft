# 선정된 시간 보정과 F/FLAT의 전체 비용

이 문서는 `horizon_decoder_diagnostic_v1`의 실제 24fit/264후보/528 FIT-TUNE/24선정 DEV가 완료된 뒤 고정했다. [과학 결과](runs/horizon_decoder_diagnostic_v1/RESULT.json) SHA는 `5f158382640dac74ec63cc3d1005b385c898821d3c032d51b543f2c9b27d7dac`다. 새 비용 실행 전 과학 감사 PASS와 선정24잠금·source·원본 state/coefficient SHA를 결속한다. 새 학습·checkpoint/alpha/arm 선정은 하지 않는다. 같은 목적의 지속 실행 범위이며 승인 대기가 아니다.

자료2(Robin/Beijing) × 입력공간3(Raw C/PCA baseK/PCA 2K) × 미래2(F/FLAT) × 출력3(기존 baseline/선정 STATIC_DELTA/선정 RAMP_DELTA)의 총36경로를 보존한다. 자료별18경로를 각각 B1/B8로 측정해 **72셀·360표본**을 남긴다. warmup2·실측5를 사용하며 사전 order seed20261015의 순환 배치를 등록한다. 18위치에서5회전은 완전한 위치 균형이 아니고 같은 첫B TUNE origin 반복이다. 독립 표본 CI나 실제 모든 온라인 입력의 지연시간이라고 해석하지 않는다.

측정 경계는 원 canonical physical CPU 과거·관측 mask에서 시작해 창 내부 causal fill/원 FIT scaler, 실제 E와 frozen F 또는 FLAT, 원 D baseline, 필요한 선정 correction, 물리 단위 CPU 반환·최종 CUDA sync까지다. actual F를 캐시로 바꾸지 않는다. 실제 B1/B8를 사용하고 canonical32로 padding하거나 결과 후 공차를 늘리지 않는다. 데이터·모델·계수 로드, 학습/선정, hash·reference·precheck·warmup·저장과 memory snapshot은 timer 밖이다. 실제 수치 계산에 포함된 shape/finite 검사와 host transfer는 timer 안이다.

PCA는 과학 결과의 같은 frozen E/center/D를 실제 CUDA FP32로 사용하고, F는 원 모델/quantile median/내부 정규화 계약을 유지한다. FLAT은 같은 E 과거의 latent-last 반복이며 F를 호출하지 않는다. 원 L7 D의 알려진 prefix·lag별 참조를 유지한다. correction은 과학 SOURCE와 같은 FP32 lag 차이, 고정 ramp, 선정 FP32 계수의 CPU matmul을 수행하고 원 baseline32에 더한다. 이때 필요한 CUDA E/F 결과의 host transfer도 포함한다. baseline이 필요 없는 중간량을 계산해 모든 경로를 느리게 만드는 식으로 실행하지 않는다.

Raw F baseline은 원 입력을 실제 F에 넣은 출력을 그대로 쓴다. `currentC+(F-currentC)`로 재결합하지 않는다. Raw FLAT baseline은 currentC의 반복이며 F/encoder가 필요 없다. Raw의 nonzero correction에는 등록된 원 C의 lag/past delta 특징이 필요하다. ZERO_DELTA 선정이면 coefficient matmul을 생략하고 원 baseline을 그대로 반환하는 경로를 별도 측정한다. 이를 새로운 표현 이득으로 세지 않는다.

각 경로의 물리 TUNE first8·선정 과학 canonical32 출력·origins·mask·scaler·E/D·코드·선정 계수의 결속을 확인한다. 등록된 기존 normalized atol1e-5/rtol2e-6, physical atol.002/rtol2e-5를 바꾸지 않는다. Raw identity 재결합의 보조 오차는 실패 gate로 쓰지 않고 원 raw baseline bytes를 기준으로 한다. baseline과 보정의 B1/B8 차이가 공차를 넘으면 그 경로 실패를 보존하고 독립 경로는 계속한다.

Raw F와 같은 공간·같은 F/FLAT baseline 각각을 분모로 paired-repeat ratio 및 ratio of medians를 분리한다. Raw와 PCA의 추가 계수 수는 달라 capacity match가 아니다. 동시 상주한 F·adapter·계수의 RSS/VRAM은 공유 프로세스 범위로만 보고한다. 매 경로 독점 메모리·최적 CPU/GPU backend라고 주장하지 않는다. F/state/계수/source/input의 전후 hash 보존, 타이머 원시 정수 ns·물리 출력·실패·자원 receipt를 남기고 독립 저장 산술/집계 검산을 수행한다. 독립 감사가 F/타이머 자체를 새로 재생한 것으로 바꾸어 말하지 않는다.

기존 새방향76셀은 원 native/ridge 모델의 비용으로 유지한다. 이72셀과 cohort/선정 상태가 다른 비용을 섞거나 정규화 확장5계수에 대신 쓰지 않는다. TEST·설치·큰 모델 학습·자동 Git·논문/PPT는 실행하지 않는다.
