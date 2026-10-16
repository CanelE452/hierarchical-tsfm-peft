# 선행연구와 실제 구현의 대응

확인일: **2026-10-15**. 이 문서는 [공개 계획 §2](plan/MASTER_PLAN_PUBLIC.md)의 설계 근거를 원문·공식 코드·현재 구현과 대조한 기록이다. 첨부 설계문 자체를 원논문의 주장으로 인용하지 않는다. `[확인]`은 직접 읽은 원문/코드 범위, `[추정·판단]`은 그 범위를 넘어서는 수학적·연구적 해석이다. 문헌의 성공을 이번 결과로 옮기거나, 포트 구현을 논문 전체 재현으로 표현하지 않는다. 학습·추론·비용의 실제 완료 여부는 별도 실행 영수증과 registry를 따른다.

## 1. 먼저 구분할 대상: Bolt 내부와 외부 표현

[확인] [Chronos-Bolt 공식 모델 카드](https://huggingface.co/amazon/chronos-bolt-small)는 과거 관측을 patch로 묶어 다단계 quantile을 직접 예측하는 모델을 설명한다. 이번 [gpu_runtime.py](gpu_runtime.py)의 `RowForecaster`는 `[B,L,C]`를 `[B×C,L]`로 바꾸어 각 원/잠재 시계열을 독립적인 row로 전달한다. 따라서 이 경로의 F 자체가 채널 간 attention이나 directed lag graph를 학습한다고 설명할 수 없다. 채널 정보는 외부 P/E/D 또는 직접 예측기의 역할이다.

[확인] `FrozenBolt`는 backbone parameter를 고정하고 `model(context=rows).quantile_preds`의 **0.5 quantile**만 사용한다. 입력으로 gradient를 전달하는 학습 경로는 유지한다. 별도 LoRA 경로는 저랭크 q/v 업데이트를 학습하므로 고정 P/E/D 경로와 구별한다. current anchor와 lag7 D는 **외부 원채널 decoder**이고, Bolt의 T5 내부 decoder와 다르다. 공식 quantile loss 구현은 [버전 태그 v2.3.2의 원 코드](https://raw.githubusercontent.com/amazon-science/chronos-forecasting/v2.3.2/src/chronos/chronos_bolt.py)에서 확인했다.

[추정·수학적 판단] 0.5 quantile은 조건부 중앙값을 겨냥하지만 제곱 손실의 최적 점 예측은 조건부 평균이다. 유한 2차 모멘트에서 `E[(Y-a)^2|X] = Var(Y|X) + (E[Y|X]-a)^2`이므로 둘을 동일하게 취급할 수 없다. 여러 잠재 시계열의 중앙값을 D로 섞는 연산도 일반적으로 원채널의 평균·중앙값을 보장하지 않는다. 이는 **가능한 인터페이스 제약**이며 현재 실패 원인을 확정하는 결과는 아니다. 외부 모듈을 원채널 normalized MSE로 학습해 보정할 여지는 있지만 MSE 최적성은 실험으로 확인해야 한다.

모델 pin과 실행 변형은 다음처럼 분리한다.

- Small snapshot: [`772f3d25d38aec6d914c8949dab4462e2d46f5d8`](https://huggingface.co/amazon/chronos-bolt-small/tree/772f3d25d38aec6d914c8949dab4462e2d46f5d8), 실제 기록은 [PROBE_ACTUAL_GPU.json](PROBE_ACTUAL_GPU.json). Mini/Tiny pin은 [MODEL_ASSETS.json](MODEL_ASSETS.json).
- 로컬 배포 패키지 식별은 `chronos-forecasting 2.3.2`; 읽은 설치 `chronos_bolt.py` SHA256은 `93b3af9e4521ea981c4228b1acaa252321f7a190512f956088394426182c8b8c`. 버전 문자열만으로 공식 태그와 바이트가 같다고 주장하지 않는다.
- wrapper는 직접 gradient-enabled forward, FP32, row microbatch 및 검증된 `SafeInstanceNorm` 치환을 사용한다. 가중치 고정이 공식 pipeline을 전혀 수정하지 않았다는 뜻은 아니다. [gpu_runtime.py](gpu_runtime.py)의 당시 SHA는 `c633f1ebf4eb24ea3fa2b025ffa6cea4ec3697c8e9285d971174179560596158`이다.

## 2. AdaPTS: 잠재공간 FM 어댑터는 이미 존재한다

[확인] Benechehab et al., **AdaPTS: Adapting Univariate Foundation Models to Probabilistic Multivariate Time Series Forecasting**, ICML 2025는 다변량 입력을 잠재공간으로 변환하고 각 잠재 차원에 단변량 FM을 적용하는 어댑터를 제안한다. 따라서 `C→K→고정 단변량 F→C`라는 발상 자체를 이번 신규성으로 삼을 수 없다. [공식 논문/초록](https://proceedings.mlr.press/v267/benechehab25a.html)

[확인] 공식 commit [`8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278`](https://github.com/abenechehab/AdaPTS/tree/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278)의 [adapters.py](https://raw.githubusercontent.com/abenechehab/AdaPTS/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapters.py)에서 `LinearAutoEncoder`, `MultichannelProjector.transform_torch/inverse_transform_torch`를 읽었고, [adapts.py](https://raw.githubusercontent.com/abenechehab/AdaPTS/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapts.py)의 `adapter_supervised_fine_tuning`이 예측 오차를 최적화하는 경로임을 확인했다. 선형 AE라는 클래스명만 보고 전체 방법을 재구성 전용이라고 설명하면 잘못이다.

이번 `networks.AdaPTSLinearPort`는 **patch size1·RevIN=False의 deterministic affine encoder/decoder 구성**을 이식한다. bias를 포함한 `decoder(F(encoder(x)))`이고, B0의 current anchor/lag7 D와 다르다. 직접 affine 연산 대응은 CPU 검사 범위이며, 원 논문의 stochastic/Bayesian 추론·uncertainty·전체 어댑터 집합·원 학습 루프·benchmark 결과를 재현한 것이 아니다. 공통 시간 분할·관측 채널 macro MSE·Bolt·checkpoint 선택은 이번 계약이다. [구현 상세](MODEL_IMPLEMENTATION_KO.md)

## 3. TSMixer와 TimeMixer: 앞단의 원리 포트

[확인] Chen et al., **TSMixer: An All-MLP Architecture for Time Series Forecasting**, TMLR 2023는 시간·feature 혼합을 이용한다. 공식 [arXiv v5](https://arxiv.org/abs/2303.06053v5)와 Google [고정 코드](https://raw.githubusercontent.com/google-research/google-research/5b09c22d73a9d35eb6c5d2a99b95677a45053466/tsmixer/tsmixer_basic/models/tsmixer.py)의 `res_block/build_model`을 확인했다. 이번 B2는 시간/채널 교대 residual 혼합을 가져오되, 한 block·hidden32·GELU·dropout 없음·0 초기화 residual 출력과 `L→hidden→L` 시간 MLP로 바꾼다. P 앞단은 원래 L 격자를 반환하며 F를 뒤에 붙인다. `DIRECT_B2`는 별도 H head를 사용한다. 원 TSMixer의 전체 구조·보조 정보·학습 조건·M5 점수 재현이 아니다.

[확인] Wang et al., **TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting**, ICLR 2024의 [원문 페이지](https://arxiv.org/abs/2405.14616)는 PDM의 성분별 다중 해상도 혼합과 FMM의 미래 예측 결합을 설명한다. 공식 [고정 TimeMixer.py](https://raw.githubusercontent.com/kwuking/TimeMixer/e24610583b36fdd8c76cc17a8df4e65759a5f460/models/TimeMixer.py) 및 [이동평균 분해 코드](https://raw.githubusercontent.com/kwuking/TimeMixer/e24610583b36fdd8c76cc17a8df4e65759a5f460/layers/Autoformer_EncDec.py)를 읽었다. B4는 평균 pooling `(1,2,4)`, 대칭 이동평균의 residual/trend, fine→coarse와 coarse→fine 혼합 원리를 가져온다. 앞단은 각 scale을 원 L 격자로 다시 보내는 residual fusion으로 FMM을 바꾸며, `DIRECT_B4`만 scale별 미래 head를 사용한다. 시점 축 downsampling과 C→K 채널 축소는 서로 다른 연산이다. 원 논문·공식 전체 모델의 성능 재현으로 표시하지 않는다.

## 4. 예측 결합과 조건부 보완: 신규성의 기준선

**Bates & Granger, The Combination of Forecasts (1969)**는 기본 선행연구다. [원 출판사 DOI](https://www.tandfonline.com/doi/abs/10.1057/jors.1969.103)는 이번 도구에서403이어서 본문 전체를 읽었다고 주장하지 않는다. 이번 전역 `g+α(b−g)`와 α `{0,.5,1,M_cal-fit}` 비교는 오래된 결합 원리를 쓰는 대조이며, 특정 정규화 macro 가중·[0,1] 제한·M_cal 닫힌형 적합이 원문과 완전히 같다고 주장하지 않는다. 원 수식과 구현 계약은 [COMBINATION_IMPLEMENTATION_KO.md](COMBINATION_IMPLEMENTATION_KO.md)에 분리되어 있다.

[확인] Sen, Yu & Dhillon, **Think Globally, Act Locally (DeepGLO)**, NeurIPS 2019 [원문 §5.2, Algorithm4](https://proceedings.neurips.cc/paper_files/paper/2019/file/3a0844cee4fcf57de0c71e9ad3035478-Paper.pdf)는 global TCN-MF 예측을 local TCN의 추가 covariate로 제공한다. 이번 H가 예측값을 조건으로 쓰는 데 연결되는 선행 원리다. 이번 구현은 `g + H(past-current,g-current,b−g)`이며 frozen 부모와 별도 M_cal을 쓴다. DeepGLO의 TCN-MF·학습 절차·원 규모·LeveledInit을 포트한 것이 아니다. 같은 용량의 실제 과거 대조, fitted alpha와 Raw-conditioned 대조가 없으면 미래 예측값의 추가 효과를 분리할 수 없다.

[확인] Lee et al., **Lightweight Online Adaption for Time Series Foundation Model Forecasts (ELF)**, ICML 2025의 [공식 초록](https://proceedings.mlr.press/v267/lee25ag.html)은 보조 forecaster와 weighter로 FM 예측을 온라인 feedback에 적응시키는 접근을 설명한다. 이는 FM+보조 예측기+결합기라는 넓은 발상의 가까운 선행연구다. 이번은 정적 A/M/V/T 계약이며 온라인 정답 feedback·ELF optimizer·원 실험을 구현하지 않았다. 초록 확인을 전체 원문/코드 재현으로 확대하지 않는다.

## 5. 과거 전용과 prefix causal은 같은 말이 아니다

[확인] 현재 `ForecastAdapter`는 입력 `[o−L,o)`만 받는다. `LAG_CAUSAL`은 `(0,1,2,4,8,16,32)`의 supplied-window 지연을 사용하고 추가 L+32 행을 읽지 않는다. 이 경로는 B0 앞단으로 제한한다. `LAG_BIDIR`은 창 안에서 뒤쪽에 놓인 과거 row도 사용하며, B2/B4 역시 전체 과거 창과 대칭 처리에 접근한다. 따라서 이들을 시점별 prefix-causal 표현이라고 부를 수 없다. [실제 구현](networks.py)

[추정·판단] 창 안의 `t+δ<o`를 보는 것은 **그 내부 시점 t의 인과 표현 조건**을 위반할 수 있지만, 곧바로 `[o,o+H)` 미래 정답 누출을 뜻하지 않는다. 실제 target 누출은 데이터 경계·정규화·checkpoint 선택·학습/추론 입력을 별도 감사해야 한다. 지연 상관이 높다는 것만으로 directed causal graph나 다른 채널의 조건부 예측 필요성이 증명되지 않는다.

## 6. Ordered prefix: 선행 원리와 아직 검증할 효과

[확인] Rippel et al., **Learning Ordered Representations with Nested Dropout**, ICML 2014 [공식 초록](https://proceedings.mlr.press/v32/rippel14.html)은 nested hidden-unit 제거로 표현 좌표를 순서화한다. 제시한 PCA 관련 이론은 특정 semi-linear AE 조건의 결과이며 현재 Bolt 예측·lag7 D에 그대로 적용되는 보장이 아니다.

[확인] Kusupati et al., **Matryoshka Representation Learning**, NeurIPS 2022의 [원문 §3·식1](https://arxiv.org/html/2205.13147v4#S3)은 prefix별 과제 손실을 가중 합쳐 학습한다. 이번 ordered pilot은 여러 K를 update별로 순환하며 원채널 미래 MSE를 최적화한다. 원문의 모든 prefix 손실을 매 update에 합하는 학습이나 분류/검색 benchmark를 그대로 이식하지 않았다. [공식 게재 페이지](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html)

[확인] 현재 코드는 실제 `E[:,:K]`, `D[:,:K,:]`와 K개 F stream을 사용한다. 단순히 C개 추론 후 일부만 잘라 K 비용으로 보고하는 경로와 다르다. 그러나 이 **구현 사실만으로 TSFM 연구 기여가 성립하지 않는다**. 독립 K, max-K-only 사후 절단, nested PCA와 같은 조건에서 실제 오차·전체 학습량·K별 노출·저장량·전환/배포 비용을 비교해야 한다. 동일 총 update가 동일 K별 노출이나 동일 GPU 시간이 아니다. 독립 K=C identity 초기화와 shared full-PCA 초기화 차이도 보존한다. [ORDERED_PROTOCOL_KO.md](ORDERED_PROTOCOL_KO.md)

## 7. LoRA: trainable 수와 실제 배포 비용을 분리

[확인] **LoRA: Low-Rank Adaptation of Large Language Models**의 [원 논문](https://arxiv.org/abs/2106.09685)과 [PEFT 공식 설명](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)은 frozen weight에 저랭크 업데이트를 적합하고 병합할 수 있음을 설명한다. 이번 `LoRAQvAdapter`는 실제 q/v leaf linear에 rank8·alpha16·dropout0 업데이트를 붙인다. 공식 PEFT 전체 구현을 호출하는 경로가 아니며 원 LLM 실험 재현도 아니다.

[확인] 비용 실행기는 병합 전/후 실제 예측 parity와 원 base의 hash 복원을 검사하도록 구현되어 있다. 따라서 hook을 유지한 LoRA만 상대하면서 외부 모듈이 빠르다고 주장하면 안 된다. 병합해도 원 C별 F 입력은 남고 외부 K 표현에는 P/E/D 비용이 추가된다. trainable parameter 감소나 C/K 비율로 whole latency·activation memory 이득을 대신하지 않는다. 검사가 구현되었다는 사실과 실제 GPU 측정 완료는 다르다. [COST_PROTOCOL_KO.md](COST_PROTOCOL_KO.md)

## 8. 재구성이 예측 적합성을 보장하는가

[추정·수학적 판단] 입력 재구성 오차가 작다는 사실만으로 압축 표현이 미래 예측에 적합하다는 보장은 없다. 예를 들어 서로 독립인 과거 성분 `X=(U,S)`에서 U 분산이 훨씬 크고 미래 목표가 `Y=S`이면, 한 차원 재구성 최적 투영은 U를 우선 보존하면서 미래에 필요한 S를 버릴 수 있다. 이는 목적함수가 다름을 보이는 반례이며 현재 자료의 실패 원인을 증명하는 실험은 아니다. 정확히 가역적인 표현과 손실이 있는 작은 K 압축도 구별해야 한다.

AdaPTS 공식 코드에는 재구성 학습과 예측 목적 적합이 모두 있으므로 “AE 이름이니 예측 학습을 하지 않는다”는 비판은 성립하지 않는다. 지금 필요한 판단은 원채널 예측 손실·같은 decoder와 predictor·같은 정보/노출 조건에서 **표현 손실, F의 잠재 미래 예측, D/readout 중 어떤 단계가 남는가**다. 이 문헌 정리가 대조학습이나 동적 graph의 필요성을 확정하거나 새 학습을 승인하는 것은 아니다.

## 9. 코드 pin·열람 범위와 신규성 유보

포트 근거의 원 URL·commit·파일 SHA는 [LITERATURE_MODEL_PINS.json](LITERATURE_MODEL_PINS.json)에 있다. 이 문서에서 고정 raw URL을 직접 열었으며 별도 로컬 원문 코드 cache가 존재한다고 가정하지 않았다.

```text
AdaPTS      8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278
 adapters.py  117c143be5d4d5a71abf34d354568997dbde0ff84d1b566ff750df43cc2ac88f
 adapts.py    75480b27e792921469ac1f0499ed49c9ade58c954ff3db68f701034951fb4dc2
Google      5b09c22d73a9d35eb6c5d2a99b95677a45053466
 tsmixer.py   84a7451f86b40b989f7ab7aa36368f24f39610ad62d7ce43e474a30a4e006159
TimeMixer   e24610583b36fdd8c76cc17a8df4e65759a5f460
 TimeMixer.py 817d62f4aac54c8566e560f6d3785856e31c8ee51460279ee3d0a4823f11d4be
 Autoformer_EncDec.py
              48745b4bb647355e9845792a855df9c59fd7df7fcc664c765351fec390c4073e
```

위 SHA는 기존 취득 영수증 값이며 이번에는 원 파일을 새로 내려받아 바이트 감사한 것이 아니다. 당시 읽은 현재 `networks.py` SHA는 `787f9caf91f32a001f929f324c7bcb570b9270331f6f60f0f56872529f86dc54`, pin JSON SHA는 `2ae4b7690bc0be68dd5273a7ff32860950376fca454400d3f3cac2bd6f3e5530`이다. 공식 파일의 commit과 현재 포트의 source hash는 다른 식별자다.

원문 접근 범위는 AdaPTS/TSMixer/TimeMixer의 공식 페이지와 위 코드, DeepGLO §5.2, MRL §3·식1, Nested Dropout·ELF 초록, LoRA 원문 초록/공식 개념 문서, Bolt 모델 카드·quantile loss 코드다. TSMixer OpenReview는 bot challenge여서 저자 arXiv v5로 확인했다. DeepGLO PDF는 첫 timeout 뒤 공식 링크로 접근했고, MRL PDF 재조회 timeout은 arXiv v4 HTML로 보완했다. Bates–Granger 본문403을 우회하거나 그 본문 열람을 주장하지 않았다. 전체 논문·공식 실험의 포괄 감사는 수행하지 않았다.

[추정·판단] 현재 이름 붙일 수 있는 것은 **기존 원리의 명시적 변형과 검산 가능한 비교 설계**다. 단순 압축·외부 어댑터·예측 결합·prefix 절단을 각각 새 기여라고 선언하지 않는다. 가능한 좁은 연구 주장은 시간 의미를 보존한 작은 K 입력이 고정 F와 맞는 조건, F가 직접 예측/같은 정보 보정에 더하는 실제 이득, 그리고 정확도–비용의 한계를 실험으로 분리하는 것이다. 신규성·일반성·완성된 방법론 판단은 이 대조와 실제 결과가 갖춰질 때까지 유보한다.
