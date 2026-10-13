# FORECAST 실패 진단과 예측 성분·보정 실험

사용자 승인: 2026-10-13 현재 대화의 지속 목표. 기존 FORECAST 실패를 좁힌 뒤, 다변량 과거에서 동결 TSFM이 예측할 소수 성분을 학습하고 나머지를 보정하여 정확도 또는 비용에 이득이 있는지 실제 분석·실험한다. 이전 캠페인을 재개하지 않는다.

## 판단 질문과 완료 조건

1. 저장된 FORECAST 원래 head / 최종 checkpoint / TUNE 선택 checkpoint / 같은 표현의 새 ridge를 실제 평가한다. 학습 표본 수, 실제 관측 범위, 선택 기준을 구분한다.
2. 동일 causal encoder와 linear decoder의 재구성 학습(REC)과 동결 Bolt를 통과한 미래 예측 학습(PRED)을 비교한다. PCA는 단순 기준이며 구조가 다른 PCA 대비 차이를 학습 목적만의 효과로 주장하지 않는다.
3. 소수 성분 경로 단독, 원자료 보정기 단독, 원자료 Bolt+동일 보정기, 압축 Bolt+동일 보정기 및 과거 재구성 잔차를 입력받는 보정기를 실제 비교한다.
4. 동일 E/D에 전체 512 잠재 과거를 보는 학습 가능한 공유 선형 예측기와 persistence를 연결하여 Bolt의 추가 가치를 검사한다. 동일 규칙의 reconstruction D를 다시 맞춘 진단으로 E 개선과 D 보정 효과를 구분한다.
5. 원수치(MSE/MAE, normalized/physical, 채널별), seed별/시간 블록별 paired 차이, 학습 시간, 전체 추론 시간·메모리를 기록하고 한국어 보고서와 그림을 만든다. 독립적인 계산/누출/동결/경로 검증을 통과해야 완료한다.

## 범위와 데이터 계약

- 실제 시작 HEAD: `837c20e9639de8c3a56419e8eddcee2b63006c34`.
- Jena(C=21), Robin(C=17)의 기존 bounded TRAIN/VAL 자산과 SHA 검증을 재사용한다. 보호 TEST는 audit hook으로 차단한다.
- 기존 TRAIN의 앞 80% FIT, 뒤 20% TUNE; 정규화는 FIT만. 미래 target은 각 split 안에 완전히 포함하고, TUNE/VAL 입력은 그 원점 이전 관측만 사용한다.
- 과거 512 / 미래 48. Stage B FIT 256, TUNE 64, VAL 기존 전체 origins(Jena371/Robin30). FIT origins는 기존 legal pool에서 균일 추출하여 기존 objective의 Robin2/Jena37 제한을 피한다. 중첩 창을 독립 표본이라고 주장하지 않는다.
- VAL은 이미 여러 개발 실험에서 사용되었으므로 이번 결과는 개발 진단이며 독립 TEST 일반화 확인이 아니다. VAL로 방법·checkpoint·lambda를 고르지 않는다.
- Stage A는 과거 결과 재사용과 재평가를 구분한다. Stage B는 신규 학습·신규 추론이다.

## 선택과 근거

- 백본: 로컬 pinned `amazon/chronos-bolt-small`, float32, 가중치 고정, median의 앞48. 이전 실험과 연결하고 대형 모델 재학습을 피한다. 학습 시에는 frozen model forward의 입력 gradient만 사용한다.
- K: `floor(C/4)`, Jena5 / Robin4. 실제 K<C인 한 압축률에서 질문을 시험하는 제한된 진단이다. 최적 K 탐색이나 전 압축률에 대한 결론은 아니다.
- E: FIT PCA로 초기화한 시점별 선형 투영 + 작은 causal convolution 보정(커널3, dilation1/2/4/8/16/32, hidden16, RF127). 출력에는 시점별 LayerNorm을 넣지 않는다. REC/PRED는 seed별 동일 초기화·배치 순서·구조를 사용한다.
- D: 동일 linear K→C, PCA 역투영 초기화. REC에서는 과거 재구성, PRED에서는 실제 Bolt 출력의 미래 예측 손실로 E/D를 공동 학습한다. 따라서 주비교는 E/D 전체 경로의 효과이며 E만의 인과 효과가 아니다.
- Loss: 관측값의 FIT 정규화 channel-macro MSE. Adam lr1e-3, batch4, 최대600 updates, gradient clipping1. checkpoint0/100/200/400/600 중 TUNE 물리단위 channel-macro MSE로 선택. REC도 Bolt 이후 미래 TUNE 기준을 동일하게 적용한다. 같은 update가 같은 계산량은 아니므로 실제 비용을 기록한다.
- 보정기: 과거512의 16시점 블록 평균32개와 마지막 값(33*C 특징)을 입력받는 joint ridge. 모든 경로에 같은 구조, FIT origins, 관측 mask, lambda `[1e-4,1e-2,1,1e2,1e4]`, TUNE 선택을 적용한다. 가중치는 각 base의 FIT 예측 잔차에 따로 적합한다.
- 추가 단순 대조: 기존 E2에서 중요했던 RAW_SELF_LAG(자기 채널의 마지막33 값→ridge)를 같은 FIT/TUNE/VAL 조건으로 포함한다. 이는 관측 범위33의 별도 기준이며, 전체512의 구조 matched 보정기 비교와 구분한다. 약한 joint 기준만 이긴 결과를 일반적인 개선으로 주장하지 않는다.
- 잔차 입력 조건: 같은 구조의 ridge에 `X-D(E(X))` 과거의 블록 특징을 준다. 원자료 보정과 구조적 잔차 보정을 구분한다.
- 단순 예측기: 전체512 잠재 과거에 shared linear512→48을 적용하고, E/D는 고정한 채 미래 정답으로 최대600회 학습한다. Bolt와 동일 E/D·잠재 과거이며, native가 추가 학습을 받는다는 비용 차이를 명시한다.
- 보조 decoder: 각 E를 고정하고 같은 FIT 역사 표본·관측 mask·ridge1e-4로 pointwise reconstruction D를 적합한다. 해당 경로와 최적화된 D를 비교한다.
- 비용: 같은 device/dtype 및 batch1/8, warm-up 후7회 반복, CUDA synchronize. 실제 전처리+E+F+D+보정+역정규화 전체를 측정하며 모델 로드는 별도. 학습/추론/검산 시간을 구분한다.

## 선행 근거와 대안

- N-BEATS(ICLR2020): https://arxiv.org/abs/1905.10437 — 과거 입력을 분해하고 예측을 합산하는 설계의 근거. 동결 Bolt 호환을 보장하지 않는다.
- NHITS(AAAI2023): https://ojs.aaai.org/index.php/AAAI/article/view/25854 — 시간 해상도를 나눠 예측하는 구조의 근거. 본 실험의 채널 압축과 동일하지 않다.
- Chronos-Bolt 공식 로컬 구현/설정: 현재 pinned checkpoint와 설치 소스를 확인한다. 공식 pipeline 대비 forward parity와 실제 입력 gradient를 검증한다.
- 후보 A: 재구성 압축→Bolt. 단순하며 정보 보존 기준이 명확하지만 예측 적합성을 보장하지 않는다.
- 후보 B(실행): 실제 동결 Bolt를 통과한 미래 손실로 E/D를 학습한다. 사용 예측기와 학습 예측기를 일치시키되 비용이 더 크다.
- 후보 C: CPC/JEPA 미래 표현 정렬. 학습 가능한 미래 성분을 강조할 수 있지만 teacher/붕괴/좌표 호환이라는 변수를 추가하므로 이번 핵심 대조에 먼저 얹지 않는다. 직접 예측 목적 이상의 추가 가치가 필요한 후속 후보로 남긴다.
- 변형 대조: 같은 E/D에서 Bolt를 단순 선형 예측기로 교체한다. TSFM을 반드시 유지한다는 가정 자체를 반증한다.

## 자원·보존·중단

- 새 연구 폴더와 `.cache/forecast_component_study_20261013`만 쓴다. 기존 파일/실험/실행 프로세스를 덮거나 종료하지 않는다.
- 보수적인 이번 로컬 실행 상한: GPU 사용 프로세스 wall60분, Stage A CPU wall15분, Stage B 및 검산 CPU-only wall30분, campaign wall3시간, 새 저장4GiB, process RAM/allocated VRAM8GiB. 기존 캠페인의 예산을 수정하거나 사용하지 않는다.
- finite-gradient/누출/동결 검사가 실패하면 해당 경로만 중단·보존한다. 낮은 정확도만으로 전체 결합 대조를 생략하지 않는다.
- 한 고정 설정·두 seed의 bounded 비교를 끝내고 loss/구조를 계속 바꾸지 않는다. 결과가 음성이면 그 범위를 보고한다.
- TEST, 기존 중단 실험 재개, 패키지 설치/클라우드/대규모 재학습, 자동 commit/push, 논문/PPT 작성은 하지 않는다.

## 해석 기준

PRED가 REC를 이기면 이 설정에서 예측 목적 E/D 적응의 이득이다. 결합이 보정 단독과 RAW_Bolt+보정을 이겨야 해당 압축 경로의 추가 가치가 있다. native 선형 대조가 같거나 좋으면 Bolt 유지 근거는 약하다. 정확도와 비용 모두 불리하면 이번 구성은 채택하지 않는다. 두 seed·두 데이터 및 반복 사용 VAL로 일반적 불가능성이나 보편적 우위를 주장하지 않는다.
