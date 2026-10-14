# 학습 가능한 Bi-Hash의 과거 전용 미래 표현 실험

## 목적과 답의 범위

사용자는 ExoTimer의 학습 가능한 두 hash 경계를 이용해 미래 예측에 도움이 되는 표현을 만들 수 있는지 실제 학습·추론·분석으로 판단하도록 요청했다. 단순 보조 손실이나 고정 KMeans 실험으로 대체하지 않는다. 구현 가능성, 학습 안정성, 표현의 예측 적합성, 배포 예측 오차·비용을 나눠 판단한다. 부정적인 개발 결과도 새 자료 실험을 생략하는 gate가 아니다. 국소적인 실패는 해당 경로만 보존하고 독립 경로를 계속한다.

이 연구는 ExoTimer 식 13–16에서 착안한 시계열 간 전이 실험이다. 원 논문의 시계열·텍스트·LLM 전체 재현이나 공식 구현 재현을 주장하지 않는다. 기존 결과는 반복 사용한 개발 증거이며, 새 자료의 결과도 본 계약과 선택 절차 안에서만 해석한다.

## 기존 자산과 새 실행

- 기존 `smallk_predictive_alignment_20261013`와 이전 데이터·Chronos-Bolt 체크포인트는 읽기만 한다. 중단한 캠페인을 재개하지 않는다.
- `bihash_future_alignment_20261013` 파일럿은 별도 선행 탐색이다. 200 update 보조 정렬과 학습 없는 사후 검색이며, 학습된 양방향 memory fusion 실험이 아니다.
- 파일럿 CP·JSON은 독립 감사했으나 forecast NPZ가 없어 오차를 재채점하지 못했다. negative 중복·미래 창 중첩, 동일 TUNE 선택·보고, GPU fixture 미계측을 기록한다. 해당 보고서의 강한 불가능성 판단을 채택하지 않는다.
- 이번 모든 새 코드·결과는 이 폴더와 동일 이름 cache 아래 fresh tag에 저장한다. 기존 결과를 덮지 않는다. 자동 commit/push, 보호 TEST, 대형 백본 재학습은 금지한다.

## 데이터 계약

모든 자료는 과거 길이 L=512, 미래 H=48, K=max(1,floor(C/4))다. 주지표는 관측된 값의 채널 균등 정규화 MSE이며, 정규화 MAE와 원단위 오차를 함께 보고한다. 학습, checkpoint 선택, 방법 선택을 주지표에 맞춘다.

Jena·Robin은 기존 TRAIN/DEV_VAL 분리를 보존하되, 저장된 normalized NPZ를 기존 통계로 원단위 float32 값으로 역변환한 다음 이번 FIT 구간에서 통계와 PCA를 새로 계산한다. 저장된 원본 normalized 값의 역변환이라는 정밀도 한계를 기록한다. FIT 256, TUNE 64를 사용하고 DEV_VAL을 방법 선택에 쓰지 않는다. 기존 실험의 수치 재현을 주장하지 않는다. 이미 여러 번 확인한 자료이므로 독립 일반화 검증으로 표현하지 않는다.

새 자료는 UCI Individual Household Electric Power Consumption(235, 채널 7, 시간별 관측 평균)과 SML2010(274, Date/Time/Day-of-week 제외 수치 채널 21, 15분)이다. 기존 저장소 검색에서 사용 흔적은 확인되지 않았다는 수준으로 표현한다. 전체 사용 이력의 부재를 증명한 것으로 해석하지 않는다. 원 자료 다운로드는 개발 단계 METHOD_LOCK 후에 시작한다.

새 자료는 시간순 60/20/20(floor) FIT/TUNE/EVAL 분리, FIT만으로 float64 통계 계산 후 float32 사용, 표준편차 <=1e-6은 1을 사용한다. 각 과거 512 창은 창 내부의 과거 관측만 forward-fill하고 시작 결측을 FIT 평균으로 채운다. 미래 target은 채우지 않고 관측 mask를 보존한다. 각 채널 미래 48 창에 적어도 한 관측이 있는 origin만 사용한다. 값 크기나 성능으로 origin을 고르지 않는다. FIT 256/TUNE 64 균등 origin, EVAL은 stride 48의 모든 합법 origin이다. 입력 창 중첩으로 표본 독립을 주장하지 않는다.

Household는 minute grid에서 중복·정렬을 검사하고 시간별 관측 평균을 낸다. 관측 한 개 이상인 채널만 관측으로 표시하며 구조적으로 불완전한 첫·마지막 hour만 제외한다. hour 종료 시점을 정보 가용 시점으로 사용하고 확인되지 않은 timezone을 UTC로 가정하지 않는다.

SML2010 두 원 파일을 UTC로 병합하고 동일 timestamp/동일 값·mask 중복만 접는다. 상충하는 중복은 해당 데이터 경로를 멈춘다. 완전한 15분 grid로 재색인하며 시간 보간은 하지 않는다. 실제 행 수·결측·중복 수를 계약에 기록한다.

원 EVAL 자료의 다운로드·파싱·분리 저장은 ingestion으로 명시한다. 학습 API는 EVAL target을 읽거나 hash하지 않는다. **두 새 자료의 학습·TUNE 선택·probe·control을 모두 끝내고 global joint model lock을 만든 다음 첫 EVAL target을 연다.** 한 자료의 오류 때문에 다른 자료의 EVAL을 먼저 열지 않는다. 실패하면 필요한 자산·재개 방법을 남긴다.

## 공통 초기화·학습

- C 상태 anchor를 유지하는 동일 `ANCHOR_NONLINEAR_D`, causal trainable encoder E, nonlinear decoder D, frozen Chronos-Bolt-small을 사용한다. 엄격한 K-only 배포라 주장하지 않는다.
- PCA는 FIT 256 과거 창의 pooled covariance로 구한 초기화다. 이후 E를 학습하며 PCA만 실험하는 연구가 아니다. eigenvector 최대 절댓값 성분을 양수로 정한다.
- seed 92601/92602/92603, 실제 600 update, batch 8, AdamW lr .001/wd 0. 미래 창이 서로 겹치지 않는 같은 batch order를 모든 방법에 쓴다. 생성할 수 없으면 해당 경로의 실패를 기록한다.
- codec과 side module을 별도로 gradient norm 1로 clip한다. checkpoint 0/100/200/400/600 모두 실제 TUNE 예측 NPZ를 저장한다. TUNE 주지표 최소, 동일 값이면 가장 이른 update를 고른다.
- 미래 encoder phi: FIT의 normalized delta(Y-x_last)와 관측 mask를 받아 R=32를 만든다. 64 hidden GELU autoencoder를 400 update/batch64로 FIT reconstruction pretrain한다. 같은 dataset/seed의 동일 phi state를 모든 방법에 복제한다. core 학습에 reconstruction weight .1을 공통 적용한다. TUNE reconstruction은 진단이며 pretrain 선택에 쓰지 않는다.
- 백본 ZeroSafe normalizer는 기존 구현과 같으며 positive variance의 forward/gradient를 보존하고 variance0의 sqrt backward를 0으로 정한다. 원 수학적 미분의 유일값으로 주장하지 않는다. 실제 backbone weight SHA 불변과 encoder gradient를 확인한다.

## 주 실험 B: 과거에 관측된 미래 패턴의 학습 가능한 조회·결합

FIT 256 origin 중 균등 128개로 고정 bank를 만든다. 현재 query의 실제 미래는 forward에 넣지 않는다. 과거 창의 각 행 시점 tau에 대해 bank origin s의 target 종료 `s+48 <= tau+1`을 만족하는 항목만 조회한다. TUNE/EVAL에서 bank를 추가·갱신하지 않는다. 현재 target/missingness mask는 prediction 입력에 없다. 과거에 알려진 bank target mask는 사용할 수 있다.

S_tau=linear(E(X)_tau), P_j=phi(bank_delta_j,bank_mask_j). 서로 다른 learnable W_s/b_s, W_f/b_f를 각각 적용한다. bit32, 각 bit의 logits [I,-I] 두 범주, 학습 soft Gumbel 또는 hard straight-through, 배포 sign(I>=0→+1). temperature는 600 update 동안 1→.25다. cosine normalization eps1e-6와 retrieval softmax temperature .2를 모든 비교에 적용한다. 이는 논문의 dot/cosine 표현과 원 코드 부재 때문에 선언한 adaptation이다.

조회된 원 continuous P를 S와 concat한 2-layer GELU fusion을 E latent에 residual로 더한다. 반대 방향은 A^T S를 column mass로 나눠 P와 결합한다. 학생 Bolt 경로와 memory decoder 경로를 sigmoid(alpha), 초기 .5로 합친다. eligible memory가 없는 경우 정확히 기존 학생 경로로 돌아간다. bank 하나뿐일 때 hash score gradient가 0인 현상을 정상적으로 구분한다.

동일 fusion·decoder 용량에서 여섯 방법:

1. B_COSINE: 동일 W_s/W_f의 연속 logits에 cosine. hash군과 활성 projection 용량을 맞춘다.
2. B_TANH: noise 없이 tanh 학습, 배포 sign.
3. B_GUMBEL: soft Gumbel 학습, 배포 sign. 주 Bi-Hash adaptation.
4. B_ST: hard Gumbel/straight-through 학습, 배포 sign.
5. B_FIXED_GUMBEL: 같은 초기 W_s/W_f를 고정한다. E/phi는 학습되므로 전체 partition이 불변이라는 주장은 하지 않는다.
6. B_UNIFORM: eligible memory 균등 조회, 동일 fusion 용량.

B는 main forecast MSE+.1 REC로 공동 학습한다. memory encoder는 선택 CP의 FIT bank cache로 배포하며 memory decoder·fusion·student hash는 실제 추론 비용에 포함한다. registered·trainable·deployment-active parameter와 실제 memory bytes를 구분한다. dense 조회이므로 hash 이름만으로 subquadratic 비용이나 bit packing 절감을 주장하지 않는다.

## 보조 실험 A: 학습에서만 미래 정답을 쓰는 표현 정렬

일반 학생 예측 경로를 유지한다. S 마지막+시간 평균과 current FIT future phi를 paired symmetric InfoNCE(각 batch8, CE/ln8, cosine/.2)로 맞춘다. InfoNCE는 ExoTimer 공식 loss라는 주장이 아닌 본 연구의 추가 목적이다. 비슷한 미래 패턴을 false negative로 만들 수 있음을 기록한다.

A_CONTINUOUS(동일 W_s/W_f 연속 logits), A_BIHASH_GUMBEL, A_BIHASH_SHUFFLE(비중첩 target의 명시적 bijection), A_BIHASH_DETACH_E(auxiliary E gradient 차단), A_BIHASH_FIXED_MAPS를 비교한다. aux weight .03/.1을 개발 TUNE의 동일 dataset-seed 그룹 mean log MSE ratio로 선택한다. A 미래/hash heads는 배포에서 제거하며 비용에 포함하지 않는다. DETACH의 baseline codec update parity는 별도 fixture로 검사한다.

## 순서와 전체 실행 목록

1. HEAD/dirty/실행 작업/데이터/model 존재·SHA preflight. 선행 파일럿 독립 감사와 기존 자산 보존 목록 작성.
2. CPU 계약 검사 → 실제 Bolt GPU 누출·gradient·update·복원 fixture → 실제 짧은 profile. profile은 600 update 완료로 세지 않는다.
3. Jena/Robin 3 seed: NO_ALIGN1+B6+A5×weight2=17/group, **102 core fits**. 각 core600. phi 6 fits400.
4. TUNE만으로 A weight와 주 B 방법을 고정하고 source/protocol METHOD_LOCK. 부정적 DEV_VAL 결과로 fresh 실험을 생략하지 않는다.
5. fresh2자료×3seed: NO_ALIGN1+B6+A5선택weight=12/group, **72 core fits**. phi6 fits400. 전체174core,12phi.
6. 사전 고정 6 probe 조건(NO_ALIGN,B_GUMBEL,B_COSINE,B_UNIFORM,A_BIHASH_GUMBEL,A_CONTINUOUS)×4자료×3seed=72조건. Eraw와 B의 Eaug를 frozen Ridge로 동일하게 비교한다. A/NO_ALIGN의 Eaug는 Eraw alias. **108 unique fits,144 scientific outputs**. 원 C 단위 미래 delta를 예측해 원 D 의존을 제거한다. 알려진 last C anchor를 동일하게 포함한다. alpha .1/1/10/100는 TUNE 선택이다.
7. 모든 자료에서 RAW_BOLT, RAW+R, raw joint Ridge, SELF33 controls를 FIT/TUNE으로 고정한다. 두 fresh자료의 core/probe/control·SHA를 함께 global lock한 뒤 EVAL을 연다.
8. 사전 고정 B_GUMBEL과 개발 TUNE에서 선택한 B 방법의 합집합에 대해 student-only/memory-only/uniform/reset-W_s-W_f-u0/noiseless-tanh를 inference-only ablation으로 분석한다. 같은 방법이면 중복 실행하지 않는다. EVAL에서 재학습·방법 선택하지 않는다. train-soft/hard→deploy-sign gap도 기록한다. 비용 536행은 일반 배포 경로에 대한 비교이며 이 진단 변형들의 추가 추론 시간은 실행 자원에 포함한다.
9. 비용: core DEV 선택weight72+fresh72=144 경로, controls16, probes108 unique 경로. batch1/8, warm2/repeat7, CUDA sync. 전체 **536 cost rows**, 동일 artifact alias 중복 측정하지 않는다. CPU window fill/FIT norm/H2D/E/hash-memory/F/D/branch/inverse norm/D2H를 포함하고 model/cache 최초 load는 별도 wall로 보고한다.
10. 모든 CP TUNE NPZ(870), 선택 FIT/TUNE/DEV_VAL/EVAL NPZ, curve·실제 update·hash entropy/occupancy·retrieval·alpha·phi variance/rank·cost·source/data/CP SHA를 저장한다. 독립 CPU 원수치 재채점 후 한국어 MD·CSV·실제 시각자료와 실행 상태를 작성한다.

## 자원·실패·결론 gate

로컬 상한: GPU 사용 process wall14400초, CPU 감사·보고7200초, goal 시작부터 campaign wall28800초, 새 study+선행pilot 저장12GiB, process RSS/allocated VRAM8GiB. 최초 에이전트의 보수적 GPU7200초 설정은 사용자 지정 상한이 아니다. 실제 profile에서 core174와CP870만 약8199초가 예상돼, 본 core 학습 시작 전에 동일 목적·데이터·학습 횟수를 유지하면서 로컬 GPU ceiling을4시간으로 조정했다. 실제 예상은 전체 약3시간이며 초기상한·변경·실측을 별도 budget receipt에 보존한다. 선행pilot 영수증의 실제 합475.524856초를 GPU 사용에 포함한다. 별도 미계측 fixture는 누락을 명시하고60초를 **측정값이 아닌 budget reserve**로 둔다. 범위가 상한에 안 들어가면 축소를 완료라고 표현하지 않고 미실행 경로·이유·재개와 추가 자원을 보고한다.

한 프로세스만 GPU 학습에 사용하며 기존 사용자 작업을 종료하지 않는다. 자원 영수증은 terminal once-write, 실행 중 상태는 stdout으로만 보고한다. 누출·자료 오류는 해당 경로에, source drift/전체 자원 초과는 필요한 gate에 적용한다. 소스 수정 후에는 새 tag/source seal로 재실행하고 기존 실패를 보존한다.

최종 답은 원형 ExoTimer 재현 여부, 실제 미래 누출, 양 hash 경계의 실제 학습, Eraw/Eaug frozen readout 변화, native/decoder/잠재 예측의 병목, binary 대 continuous/uniform의 필요성, 새 자료와 seed의 불확실성, Chronos-Bolt 유지 근거 순서다. 검사 PASS를 성능 개선으로, TUNE 이득을 독립 검증으로, 유한한 음성 실험을 보편적 불가능성으로 바꾸지 않는다.

