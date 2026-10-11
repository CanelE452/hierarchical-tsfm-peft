# PCA 외 compressor 후보: 원형의 계약과 현재 실험의 차이

2026-10-10. 원문과 저자 공식 코드에 근거한 제한된 조사다. `[확인]`은 실제로 읽은 논문·코드의 내용, `[설계안]`은 현재 frozen scalar TSFM에 연결하기 위한 변경, `[판단]`은 그 변경의 연구적 의미다. 논문 원형의 성능을 현재 자료에서 재현했다는 뜻은 아니다. 이 조사 담당자는 학습·forward를 실행하지 않았다.

**[판단] PCA를 최종 방법으로 고정할 이유는 없다. 다만 새 compressor를 선택할 때는 ‘좋은 표현’과 ‘각 좌표를 기존 단변량 TSFM이 예측하기 좋은 표현’을 구분해야 한다.** 교수님의 제안은 시간·채널 의존성을 함께 읽는 표현을 탐색할 근거가 있다. 추천된 모델들이 별도 변경 없이 C→K, L 보존, causal export, 미래 복원까지 모두 만족하는 것은 아니다.

## 이번에 공통으로 필요한 계약

과거 입력 `X[B,L,C]`에서 `Z[B,L,K]`를 만들고, frozen TSFM은 K개 좌표를 각각 예측해 `Z_hat[B,H,K]`를 반환한다. decoder는 관측 과거 latent tail과 예측된 미래 latent로 `Y_hat[B,H,C]`를 만든다. 미래 정답 latent를 decoder 입력에 넣지 않는다.

‘동일 decoder’는 구조·시차 범위·학습 목표·자료·선택 규칙이 같다는 뜻이다. 서로 다른 encoder 좌표에 같은 decoder 계수를 강제로 공유하면 좌표 정렬이라는 별도 가정을 추가한다. 현재 PLAN의 방법별 별도 ridge 계수는 이 의미에 맞는다.

현재 신규 study는 각 ridge 입력 feature의 RMS를 TRAIN에서만 계산해 규격화하는 계약도 적용할 예정이다. latent 좌표의 크기가 다르면 같은 수치의 ridge λ도 다른 정규화 효과를 낼 수 있으므로 이 조건을 공통으로 두는 것이 타당하다. 실제 적용·완료 여부는 protocol 및 runner 기록으로 확인해야 하며, 이것만으로 서로 다른 latent 기저나 분포가 같은 것이 되지는 않는다.

교수님이 강조한 다섯 특징은 각각 확인해야 한다. 작은 feature dimension이 곧 작은 채널 수의 시계열이라는 뜻은 아니며, multivariate 입력 지원은 채널 간 관계를 식별했다는 뜻이 아니다. 시간 필터가 지연 패턴을 표현할 수 있다는 것과 directed lag graph를 발견했다는 것도 다르다. 여기서 causal은 미래 입력을 보지 않는 계산을 뜻하며 물리적 인과관계 발견을 뜻하지 않는다.

```text
후보 계열             원형의 압축/출력               L 보존 및 causal 연결       원형 복원 D
GTConvAE              시간 축소, 공간 노드 유지       C→K·L 보존은 변경 필요      있음
TS2Vec                시점별 feature 차원 d           가능; 과거종료 sliding 필요  없음
causal TCN-AE         현재 설계에서는 feature C→K     가능; 원 TCN에 AE 추가      원 TCN에는 없음
dynamic/lag AE        현재 설계에서는 feature C→K     가능; 과거 FIR 계약         현재 설계에 있음
DCA                   pointwise 선형 C→K             그대로 가능                없음
DAPC                  시점별 C→d                     가능; encoder 선택 조건     재구성 목적에 있음
CoST                  시점별 trend/season feature     sliding export 필요         없음
TNC                   다변량 창 전체→벡터             과거종료 창으로 변경 필요   없음
```

위의 ‘가능’은 입출력 계약이 성립할 수 있다는 뜻이다. frozen scalar TSFM의 예측 품질 보장은 어느 항목에도 없다.

## 1. GTConvAE: 교수님 제안의 graph-time 원리

**[확인]** 원형은 주어진 공간 그래프 S와 시간 연산 S_T의 product graph에서 공동 필터링한다. p3 Eq1의 학습 가능한 네 결합 계수는 채널쌍마다 미지의 lag edge를 학습하는 변수와 다르다. p4 Eq4·Eq6은 시간 downsampling 후에도 공간 노드를 유지하며 latent는 `T_e × N × F` 형태다. p5 Eq7–10의 decoder는 시간 upsampling으로 복원한다. 재구성 목적과 multivariate graph-time 표현을 제공하지만 현재의 `L × K` 채널 병목과 원형 latent 축이 다르다. [논문](https://proceedings.mlr.press/v198/sabbaqi22a.html)

**[설계안]** 시간 pooling을 생략하고 causal graph-time 필터 뒤에 C×node-feature를 K로 읽는 projection을 두면 현재 계약에 맞는 nonlinear compressor를 만들 수 있다. decoder는 공통 D8로 교체할 수 있다. 이는 GTConvAE-inspired 설계이며 원형 재현도 원형의 안정성 정리 적용 확인도 아니다. TRAIN 동시점 절대 상관 top3 graph는 지연·방향성 인과 그래프 발견의 결과가 아니다.

**[실행 구분]** 앞선 pilot의 graph-time 선형 변형은 실행됐다. 이번 GRAPH_TIME_NL의 완료 여부는 이 폴더의 runner 결과로 판단한다. GTConvAE 원형은 실행하지 않았다. PMLR·원문·한정 검색에서 저자 공식 저장소를 확인하지 못했다. 공식 코드가 존재하지 않는다고 단정하지 않는다.

## 2. TS2Vec: 원형 자체로 시점별 표현을 제공하는 교수님 후보

**[확인]** multivariate `[B,L,C]`를 `[B,L,d]`로 변환하며 `output_dims`를 작게 지정할 수 있다. 공식 기본값은 320이다. 서로 다른 crop/mask의 같은 시점 표현에 hierarchical contrastive loss를 적용한다. 마지막/full-series pooling은 시계열을 하나의 벡터로 만들므로 현재 경로에는 시점별 출력을 사용해야 한다. 원문의 예측은 representation→ridge→미래 원채널 값이며 재구성 decoder는 제공하지 않는다. [논문](https://ojs.aaai.org/index.php/AAAI/article/view/20881), [공식 코드](https://github.com/zhihanyue/ts2vec)

**[확인: 코드]** 기본 CNN은 양방향 문맥을 읽는다. `causal=True`는 sliding branch에서 오른쪽 문맥을 제거하며 공식 forecasting은 `sliding_length=1`을 사용한다. 큰 sliding block 안에서는 앞 시점 표현이 같은 block의 뒤 시점을 읽을 수 있다. `causal=True`만 설정하고 sliding을 생략하면 causal export가 된다고 말할 수 없다. 과거 window 전체를 양방향으로 인코딩하는 것 자체가 미래 목표 누수라는 뜻은 아니지만, rolling representation의 의미를 통일하려면 과거 종료 export를 명시해야 한다. [API](https://github.com/zhihanyue/ts2vec/blob/b0088e14a99706c05451316dc6db8d3da9351163/ts2vec.py#L206), [예측 코드](https://github.com/zhihanyue/ts2vec/blob/b0088e14a99706c05451316dc6db8d3da9351163/tasks/forecasting.py)

**[설계안]** K=5/6 시점별 출력, causal rolling export, 고정 E와 별도 reconstruction/forecast D8를 연결할 수 있다. 작은 K에서 contrastive 품질과 복원 가능성은 실험 문제다. TS2Vec objective나 cross-channel 입력만으로 time-delayed 관계 식별 또는 Bolt 호환성을 보장하지 않는다. TS2Vec 원형·이 변형 모두 이번까지 실행하지 않았다.

## 3. nonlinear causal TCN-AE: 입출력 변경을 작게 하는 후보

**[확인]** TCN 원문 §3(p3–4)은 causal convolution, 길이 보존, dilation과 residual block을 설명한다. 공식 `TCN/tcn.py`의 convolution·Chomp는 출력 길이를 유지하면서 미래 입력을 제거한다. 단, 이 논문은 AE compressor 논문이 아니다. [원문](https://arxiv.org/abs/1803.01271), [공식 TCN 구현](https://github.com/locuslab/TCN)

**[설계안]** pooling 없는 causal CNN으로 `C→hidden→K`를 수행하고 공통 D8로 reconstruct하면 unsupervised nonlinear temporal AE가 된다. 일반 convolution이 입력 채널들을 함께 읽으므로 cross-channel lag 조합을 표현할 수 있다. 실제 자료에 그 패턴이 필요하다는 증거는 별도다. 이번 TCN_TIME은 PLAN상 작은 kernel9 causal convolution+tanh+K projection이다. full dilated residual TCN 원형의 재현이라고 부르지 않는다.

**[판단]** 동일 decoder·목표를 유지하고 가까운 변수 수의 MLP_POINT와 비교하면 선형 lag 대조만보다 temporal 입력의 유용성을 평가하기 좋다. 이는 단순한 후보 설계이며 새 방법의 기여가 자동으로 생기는 것은 아니다. 이번 변형의 실제 완료 상태는 result receipt를 따른다.

## 4. dynamic-factor / lag AE와 TRMF-inspired 목적

**[확인: 동적요인]** GDFM 2005 §2 Eq2.1은 `x_t=B(L)f_t+ξ_t`로 같은 요인이 서로 다른 지연 계수로 관측되는 구조를 다룬다. 동적 요인 수와 동시점 행렬 rank는 같을 필요가 없다. 원형은 large-panel stationarity·spectral 조건과 별도 요인 추정·예측 절차를 사용하므로 소수 채널의 학습형 AE와 동일하지 않다. [저자 원고](https://lippi.ws/SC/Forecasting.pdf)

**[설계안]** `z_t=Σ_l A_l x_(t-l)`, `x_hat_t=Σ_j B_j z_(t-j)` 형태의 causal FIR E/D는 L을 유지한 C→K compressor다. PCA-free 선형 AE 또는 nonlinear lag AE를 만들 수 있고, A의 0시점 tap만 사용하는 pointwise 대조가 가능하다. E/D의 잠재 공간이 저차원이라는 사실만으로 물리적 요인·지연을 회복했다는 주장은 할 수 없다. GDFM 원형은 미실행이며 기존 pilot에서는 PCA 초기화 lag AE 변형만 실행했다. 이번은 PCA-free 대조다.

**[확인: TRMF]** TRMF §4 Eq9–10은 reconstruction에 latent별 AR temporal regularization을 결합한다. 공식 구현은 잠재 시계열 자체와 loading을 최적화하고 latent AR로 미래를 생성한다. 새로운 과거 window를 바로 인코딩하는 amortized E는 원형의 계약이 아니다. [논문](https://www.cs.utexas.edu/~inderjit/public_papers/trmf_nips16.pdf), [저자 코드](https://github.com/rofuyu/exp-trmf-nips16)

**[설계안·판단]** AE에 latent별 자기 과거 예측 손실을 더하는 TRMF-inspired 후보는 단변량 predictor에 맞는 좌표를 유도할 단서다. latent scale 축소·decoder scale 확대에 대한 통제와 간단한 AR만으로 충분한지 확인이 필요하다. AR 예측성이 frozen Bolt 호환성을 뜻하지는 않는다. 이번 matched study에는 이 목적을 추가하지 않았고 TRMF 원형도 미실행이다.

## 5. DCA: 분산이 아닌 공동 예측정보로 선형 subspace 선택

**[확인]** DCA는 pointwise projection `z_t=Vᵀx_t`로 C→K와 L 보존을 바로 수행한다. 학습은 과거–미래 window의 Gaussian predictive information을 최대화하며 cross-lag covariance를 포함한다(p3–4, Eq1–3). inference는 pointwise이므로 별도 미래 문맥이 없다. decoder는 원형 구성에 없다. 공식 `transform`은 `(X-mean_) @ coef_`다. [NeurIPS 원문](https://papers.nips.cc/paper_files/paper/2019/file/704cddc91e28d1a5517518b2f12bc321-Paper.pdf), [공식 코드](https://github.com/BouchardLab/DynamicalComponentsAnalysis)

**[판단]** PCA와 같은 선형 E 형식에서 선택 목적만 바꾸는 후보다. DCA의 temporal objective는 encoder가 매 시점 과거 입력을 추가로 읽는다는 뜻은 아니다. stationary/Gaussian approximation과 covariance 추정이 필요하며 현재 nonstationary 자료의 효용은 미확인이다.

원문 p4는 PI가 latent의 가역 선형 변환에 invariant하다고 명시한다. 따라서 공동 subspace의 예측정보가 커도 좌표별 frozen scalar TSFM에 적합한 기저를 정해주지는 않는다. 별도 D8와 실제 미래 평가가 필요하다. 원형·현재 연결 변형 모두 미실행이다.

## 6. DAPC: predictive information과 reconstruction을 함께 학습

**[확인]** DAPC는 nonlinear 시점별 encoder로 얻은 latent의 predictive information에 masked reconstruction과 covariance penalty를 결합한다(p2–4, Eq1–5). p2 각주2는 이 연구의 latent/input 길이가 같다고 명시한다. decoder는 latent에서 관측을 복원하는 feed-forward network다. 원문은 PI만 최적화하면 강한 encoder가 입력 정보를 잃을 수 있다는 문제를 직접 다룬다. [원문](https://arxiv.org/abs/2010.03135), [저자 코드](https://github.com/JunwenBai/DAPC)

**[확인: 코드]** 공식 코드는 linear/DNN/GRU/LSTM 및 bidirectional/Transformer 선택을 제공한다. `solver.py`의 unidirectional GRU/LSTM은 시점별 sequence를 반환한다. 미래를 읽지 않는 입력 전처리와 이 encoder를 선택하면 causal export의 구현 근거가 있다. bidirectional/일반 attention 원형 설정까지 causal이라고 부르면 안 된다. 저장소는 주로 Lorenz 실험을 제공하며 본 조사에서는 환경 재현을 하지 않았다.

**[판단]** 단순 reconstruction보다 latent dynamics를 고려하는 AE 목적이라는 점에서 직접적인 후속 후보다. Gaussian PI는 latent 공동 예측성이고 covariance identity penalty도 scalar FM 적합성을 보장하지 않는다. masked/shifted reconstruction 및 PI를 이번에 동시에 추가하면 encoder 시간 효과의 원인 분리가 어려워지므로 현재 대조를 바꾸지 않는다. 원형·연결 변형 모두 미실행이다.

## 7. CoST: trend/season 분해와 contrastive 학습

**[확인]** CoST는 multivariate `[L,C]→[L,d]` representation을 정의하고 마지막 시점 표현으로 ridge forecasting을 수행한다(p2, §3). causal trend filters와 frequency-domain seasonal filtering에 별도 contrastive loss를 적용한다. reconstruction decoder는 제공하지 않는다. [논문](https://arxiv.org/abs/2202.01575), [공식 코드](https://github.com/salesforce/CoST)

**[확인: 코드]** Fourier branch는 입력 창 전체를 읽는다. 공개 forecasting export는 각 창의 마지막 trend/season feature를 합친 뒤 `casual=True, sliding_length=1`로 시점별 sequence를 만든다. `casual`은 공식 API의 철자다. 전체 창의 모든 intermediate 시점이 causal하다고 간주하면 안 된다. `output_dims//2`씩 두 component를 사용하므로 홀수 d=5를 지정하면 합친 차원이 4가 된다. K5 병목은 별도 projection/분할 변경이 필요하다. [encoder](https://github.com/salesforce/CoST/blob/7bb5271c2b66cd5a1d634c27f5df308f190b50ec/models/encoder.py), [export](https://github.com/salesforce/CoST/blob/7bb5271c2b66cd5a1d634c27f5df308f190b50ec/tasks/forecasting.py)

**[판단]** 계절성이 중요하다면 조사할 가치가 있지만 작은 K로 trend/season을 나누는 비용과 causal export 반복 비용을 확인해야 한다. 원문의 causal 설명은 생성모형의 가정이며 채널 간 directed lag graph 발견을 의미하지 않는다. 원형·연결 변형 모두 미실행이다.

## 8. TNC: 시간적으로 가까운 창의 상태 표현

**[확인]** TNC는 다변량 중심 창 `[C,δ]`를 단일 벡터 `[d]`로 인코딩한다(p2–4, Eq1). 가까운 창/먼 창을 구분하는 weighted neighborhood objective를 사용하며 decoder가 없다. 공식 RNN 기본값은 bidirectional이고 마지막 RNN output만 반환한다. 따라서 원형은 L×K sequence AE가 아니다. [논문](https://arxiv.org/abs/2106.00750), [공식 코드](https://github.com/sanatonek/TNC_representation_learning)

**[설계안]** 매 시점 과거 종료 window만 인코딩하면 L개의 feature sequence를 만들 수 있으나 원형 중심 window 계약을 바꾼 것이다. representation dimension을 K로 설정하고 별도 D8를 학습할 수 있다. neighborhood가 시간적 상태를 구분한다는 목표는 reconstruction·미래 예측·lagged cross-channel relation 식별을 직접 최적화하는 것과 다르다. 원형·연결 변형 모두 미실행이다.

## 직접 겹치는 선행과 현재 남는 질문

**[확인] AdaPTS(ICML2025)**는 학습형 feature-space E/D 사이에서 frozen univariate FM을 latent별 독립 적용한다. 선형·비선형 AE와 forecast objective의 adapter 학습을 이미 다룬다. 따라서 ‘PCA-free AE + frozen TSFM’ 자체를 기여로 잡을 수 없다. 공식 `adapters.py`와 `adapts.py`도 이 계약을 확인한다. [원문](https://proceedings.mlr.press/v267/benechehab25a.html), [공식 코드](https://github.com/abenechehab/AdaPTS)

**[확인] LatentTSF(ICML2026)**는 reconstruction-pretrained pointwise AE를 고정한 뒤 latent-space predictor를 학습하는 접근이다. compression과 expansion 둘 다 가능하다. 원문의 latent objective는 forecast predictor를 학습하므로 임의의 latent에 기존 frozen Bolt를 그대로 적용하는 효용의 증거는 아니다. 현재 decoder-only adaptation과 구별하되 latent forecasting을 새 아이디어로 과대포장하지 않는다. [원문](https://proceedings.mlr.press/v306/yang26y.html), [공식 코드](https://github.com/Muyiiiii/LatentTSF)

위 두 논문은 2026-10-10 온라인 상태와 공식 저장소 revision을 확인했다. 검색 범위는 직접 latent compression/FM adaptation과 representation 후보에 제한했으며 완전한 신규성 조사가 아니다. 원형들을 현재 benchmark에서 실행하지 않았다.

## 현재 단계의 후보 우선순위

1. **지금 승인된 PCA-free matched study를 먼저 읽는다.** pointwise/temporal 선형 쌍과 MLP/작은 causal CNN 쌍은 같은 D8 아래 encoder 시간 정보의 추가 가치를 묻는다. GRAPH_TIME_NL은 교수님 graph-time 제안과 연결되는 가능성 screen이며 graph만의 효과를 분리하는 대조는 아니다. 결과가 나오기 전 후보 우열을 미리 정하지 않는다.
2. **교수님 제안의 다음 직접 후보는 TS2Vec의 작은 K·causal export·별도 D 연결이다.** 원형 표현학습 objective가 달라 현재 reconstruction AE들과 보완적으로 비교할 가치가 있다. 공식 구현을 K와 현재 자료에 연결하는 수정, causal export 비용, reconstruction/forecast D를 포함해야 하므로 아직 실행한 것으로 표시하지 않는다. GTConvAE를 더 충실히 구현하는 일도 선택지지만 현재 graph-time 변형과 원형의 축·필터·pooling 차이를 먼저 남겨야 한다.
3. **표현의 학습 목적이 다음 문제라면 DCA/DAPC/TRMF-inspired 후보를 좁혀 선택한다.** DCA는 같은 pointwise 선형 축에서 시간 목적을 바꾸는 대조, DAPC는 nonlinear predictive-information+reconstruction, TRMF-inspired는 latent별 자기 과거 예측 구조를 겨냥한다. 세 목적을 한꺼번에 추가하는 제안은 아니다.
4. **CoST/TNC는 도메인 조건과 변경량을 보고 이후 선택한다.** 계절성 분해나 상태 neighborhood가 현재 실패 사례와 직접 연결되면 유용한 후보지만, 원형 출력·causal export·decoder 계약 변경이 더 필요하다. 현재 문헌 조사 결과만으로 이 모델들을 탈락시키거나 바로 대규모 학습을 권하지 않는다.

**[문제 정의 후보] 같은 K·frozen scalar TSFM·decoder 학습 계약에서, 시간/채널 정보를 읽는 encoder가 pointwise encoder보다 추가 미래 정보를 유효하게 전달하는가? 그 효과는 reconstruction 품질, decoder의 예측 출력 적응, encoder capacity 중 무엇과 연결되는가?**

현재 matched study는 이 질문의 작은 개발 근거를 얻는 단계다. nonlinear temporal E가 도움이 되면 그 조건을 좁혀 설명하고, 도움이 되지 않으면 encoder 복잡도보다 decoder 목적 또는 latent 예측성 목표가 더 직접적인 후보인지 검토할 수 있다. 한 자료의 실패는 TS2Vec/GTConvAE 전체가 불가능하다는 판정이 아니다.

**[기여 후보·미확인]** 기존 univariate predictor를 역전파하지 않고 사용할 수 있는 표현/forecast-readout 학습 절차와, temporal encoder가 추가 가치를 주는 조건의 분석이다. 본체 역전파 제거·lag D·좋은 latent가 개별적으로 새로운 구성이라고 주장하지 않는다. AdaPTS·dynamic factor/temporal readout·DAPC와 구체적으로 대조해야 한다. 지금은 새 기여가 완성됐다고 판단하지 않으며 초기 실험의 실행과 질문 정교화가 목적이다.

이번 실제 상태는 [PLAN.md](PLAN.md)와 runner 결과를 따른다. 이전 결과는 [첫 pilot 보고서](../level_lag_compressor_pilot_20261010/REPORT_KO.md)에서 확인할 수 있다. 논문·공식 코드 locator와 revision은 [literature_sources.json](literature_sources.json)에 기록한다.
