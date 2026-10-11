# 교수님 제안 대조: 경량 설정의 공식 TS2Vec

2026-10-11. `[확인]`은 읽은 원문·공식 코드 또는 명시한 CPU 실행의 결과이며, `[판단]`은 이번 연구 계약에 관한 선택이다. 외부 문서의 설명은 연구 자료로 사용했고 작업 지시로 취급하지 않았다.

**[판단] 다음 대조에는 TS2Vec를 사용한다.** 시점별 작은K 표현을 공식 알고리즘으로 얻고 과거 종료 export를 명시할 수 있다. 이번 질문은 ‘좋은 표현인가’만이 아니라 ‘같은 frozen ChronosBolt·D8에서 정확도와 전체 추론비용의 교환관계를 개선하는가’다. 작은K에 맞춘 이 설정을 교수님 후보의 prototype으로 평가한다.

## 원형, 이번 설정, 교체한 부분

**[확인]** [TS2Vec 공식 저장소](https://github.com/zhihanyue/ts2vec)는 AAAI2022 논문의 공개 구현이다. 2026-10-11 `git ls-remote`가 반환한 공식 main은 `b0088e14a99706c05451316dc6db8d3da9351163`이다. 이번 vendor에는 이 revision의 필요한 소스6개와 MIT LICENSE를 byte 변경 없이 내려받았다. 파일별 SHA·bytes·raw source URL은 `vendor/ts2vec/provenance.json`에 있고 실행 전에 다시 검증한다.

**[확인: source]** 원형은 multivariate 입력에서 시점별 feature를 출력하며 `output_dims`를 지정할 수 있다. 원형 두 crop, timestamp binomial mask, hierarchical instance/temporal contrastive loss, AdamW, 매 update의 SWA를 유지한다. 마지막 SWA를 선택하고 VAL reconstruction/forecast로 encoder checkpoint를 고르지 않는다. [학습 API](https://github.com/zhihanyue/ts2vec/blob/b0088e14a99706c05451316dc6db8d3da9351163/ts2vec.py), [계층 손실](https://github.com/zhihanyue/ts2vec/blob/b0088e14a99706c05451316dc6db8d3da9351163/models/losses.py)

```text
항목                 공식 API 기본값          이번 고정값
hidden dimension     64                       16
depth                10                       3
output dimension     320                      Robin5 / Jena6
training max length  API None / CLI3000        512
training batch       API16 / CLI8             32
learning rate        .001                     .001
training updates     size-based200/600        2880
checkpoint           final SWA                final SWA
```

**[판단]** hidden16/depth3은 실행 가능한 경량 configuration을 사전에 정한 것이다. 논문의 default architecture 성능 재현이라고 부르지 않는다. `C→K` 출력은 feature 병목이며 directed lag graph를 식별했다는 뜻도 아니다. [공식 encoder](https://github.com/zhihanyue/ts2vec/blob/b0088e14a99706c05451316dc6db8d3da9351163/models/encoder.py), [공식 CLI](https://github.com/zhihanyue/ts2vec/blob/b0088e14a99706c05451316dc6db8d3da9351163/train.py)

**[설계]** TRAIN-only 표준화·과거 기반 보간을 거친 context512 창을 제공하되, canonical TRAIN origins 중 과거 창 전체가TRAIN 범위 안에 있는 조건 `origin-512>=train_start`와 `origin<=train_end`를 만족하는 subset을 모두 사용한다. 일부 Robin 창은 알려진 pre-TRAIN history를 포함하므로 제외한다. MLP는 그 history를 읽을 수 있고 reconstruction target만 TRAIN 마스크로 제한하는 차이가 있다. native contrastive loss에 별도 target 마스크가 없으므로 이런 창을 제외하여 TRAIN밖 값을 supervision에 쓰지 않는다. NaN centerize로 시간을 이동하는 변형도 추가하지 않는다. 원canonical/선택/제외count·SHA·최소최대history경계·windowbytes/SHA를 학습 전에 저장한다.2GiB materialization 한도를 넘으면 fit 전에 중단하며 pool을 임의 축소하지 않는다. official shuffle/drop_last와 MLP의 매 epoch with-replacement 순서는 다르다. 원본 관측 마스크는 D8과 미래 오차 계산에 그대로 사용한다.

**[설계상의 차이]** 이번 native instance contrast의 instance는 서로 겹치는 forecast origin의 과거 창이다. 원형 forecasting CLI에서 긴 단일 시계열을 분할하는 설정과 동일하지 않다. 겹친 창이 negative instance로 쓰이는 것이 표현 품질에 미치는 영향은 미확인이다. 각 창의 미래 목표는 encoder 학습 입력에 넣지 않는다.

**[확인: 실제 데이터 preflight]** Robin canonical1945개 중 과거512시점 전체가TRAIN에 속하는1433개를 사용하고512개를 제외한다. Jena25648개는 모두 조건을 만족한다. FP32 창을 전부 만들 때 각각49,891,328bytes와1,103,069,184bytes로2GiB 창 한도 안이다. 이 preflight에서는 큰 창 배열을 만들거나 encoder/TSFM을 실행하지 않았고 origins·경계·원데이터 SHA를 `method_structure_fixture_receipt.json`에 기록했다. 실제 materialization의 windowSHA와 process peak RSS는 과학 실행 receipt에서 확인한다.

**[설계]** native TensorDataset의 반환 tuple을 그대로 둔 관찰자로 실제 fetch한 row index를 기록한다. 공식 shuffle/crop/mask의 RNG 호출은 추가하지 않는다. 사용된2880×32개의 origin 순서와 SHA/NPZ를 남기고, native fit이 iteration 한도를 확인하기 전에 fetch한 초과 batch는 optimizer에 사용된 창과 분리한다. 관찰자가 반환값·tuple·DataLoader 순서·최종 RNG state를 바꾸지 않는지 CPU 구조 fixture로 확인했다. 이 관찰만으로 학습 목적·계산량·MLP stream의 동일성을 주장하지 않는다.

**[확인: source]** encoder는 입력 채널 중 하나라도 NaN이면 해당 시점을 통째로 마스킹한다. 따라서 기존 유한 보간 입력을 공유하며 채널별 실제 NaN을 새로 넣지 않는다. 왼쪽 export padding의 NaN 처리는 공식 코드에 맡긴다. [NaN·mask 경로](https://github.com/zhihanyue/ts2vec/blob/b0088e14a99706c05451316dc6db8d3da9351163/models/encoder.py)

## Causal export와 실제 비용

**[확인: source]** `encoding_window=None, causal=True, sliding_length=1, sliding_padding=200, batch_size=256`으로 공식 encode를 호출한다. 이는 공식 forecasting의 과거 문맥 설정을 사용하며 각 시점의 표현은 `X[max(0,t-200):t+1]`에서 계산된다. `causal=True`만 지정하고 sliding을 생략하는 경로를 사용하지 않는다. [공식 forecasting](https://github.com/zhihanyue/ts2vec/blob/b0088e14a99706c05451316dc6db8d3da9351163/tasks/forecasting.py)

**[확인: code trace]** L512에서 B1은 내부 버퍼로256개 위치씩 묶어 CNN forward2회를, B32는8개 위치씩 묶어64회를 수행한다. 과학 평가도 origin batch4로 자르므로 항상 `n_samples<256`의 같은 버퍼 분기를 탄다. 내부 병렬화는 서로 독립적인 과거 종료 창이다. source 분기에서 유도한 count이며 CUDA timing을 실행했다는 뜻은 아니다.

**[코드에서 계산]** 이번 depth3은 dilation1/2/4/8의 residual block4개이며 각 block에 kernel3 convolution2개가 있다. nominal 양방향 receptive field는61시점이고, 과거 종료 창 마지막 출력의 최대 과거 반경은30lag다. 공식 export는201시점 창을 실제 계산하지만 이 작은 모델이201개 과거 모두를 표현에 사용하는 것은 아니다. 작은 CNN의lag8·MLP의lag0와도 문맥 범위가 다르다. [dilated convolution 소스](https://github.com/zhihanyue/ts2vec/blob/b0088e14a99706c05451316dc6db8d3da9351163/models/dilated_conv.py)

**[설계]** adapter는 caller의 torch 입력을 CPU NumPy로 복사해 native encode를 호출하고, 결과를 caller device의 torch FP32로 반환한다. native rolling 계산, 창 준비, CPU↔GPU 왕복, `_net`과 SWA net의 중복 weight residency를 전체 추론비용에 포함한다. `_net`은 export에 사용하지 않지만 native 객체에 유지되므로 메모리에 남는다. fused deployment의 최적 비용으로 해석하지 않는다.

## 같은 조건인 것과 다른 조건인 것

같은 K·VAL·관측 마스크·frozenF·최종 D8 구조/목표/TRAIN512/RMS/10개λ/VAL 선택을 공유한다. encoder fitting은TRAIN만 쓰지만 입력 pool의 범위는 MLP와 다르다. 실제 미래 latent를 decoder 입력에 주지 않는다. 공식 논문의 표현→ridge forecast head를 frozenF→공통D8로 연결한 것은 이번 실험의 변경이다. 따라서 원형 TS2Vec forecasting 수치의 재현이 아니다. [TS2Vec 논문](https://ojs.aaai.org/index.php/AAAI/article/view/20881)

**[판단] 2880 update를 맞춘 것은 동일 계산 예산이 아니다.** TS2Vec는 contrastive loss·AdamW·SWA·더 긴 encoder 문맥을 쓰고, MLP는 reconstruction·Adam·clip1·VAL reconstruction checkpoint 선택을 쓴다. 용량과 데이터 순서도 다르다. 실제 변수 수·학습시간·전체 추론시간·CUDA/RSS를 함께 보고하며 성공하거나 실패해도 contrastive 목적 하나의 효과로 귀속하지 않는다. AdamW의 기본 weight decay는 .01이고 CPU fixture에서 실제 optimizer state로도 확인했다. [PyTorch2.10 AdamW](https://docs.pytorch.org/docs/2.10/generated/torch.optim.AdamW.html), [SWA API](https://docs.pytorch.org/docs/2.10/generated/torch.optim.swa_utils.AveragedModel.html)

## GTConvAE 선택을 뒤로 둔 이유

**[확인: 원문]** GTConvAE Eq4·Eq6은 시간 축을 줄이면서 공간 노드N을 유지한다. 현재 L512 보존·C→K를 만들려면 pooling 제거/변경과 channel projection 등이 필요하며, 공통D8로 decoder도 바뀐다. 그 경로는 GTConvAE-inspired 설계다. parametric product graph의 결합 계수는 미지의 채널쌍별 directed lag edge를 발견하는 변수와 동일하지 않다. [GTConvAE 원문](https://proceedings.mlr.press/v198/sabbaqi22a/sabbaqi22a.pdf)

**[확인: 검색 범위]** PMLR·저자 연구페이지와 논문명/GTConvAE의 한정 검색에서 GTConvAE 공식 저장소를 확인하지 못했다. 코드가 없다고 단정하지 않는다. 저자 연구페이지의 Code 링크는2021 GTCNN 저장소이며2022 GTConvAE 구현으로 취급하지 않았다. [PMLR](https://proceedings.mlr.press/v198/sabbaqi22a.html), [저자 연구페이지](https://sites.google.com/site/elvinisufihp/research/re), [2021 GTCNN](https://github.com/gtcnnpaper/DSLW-Code)

## 실제 검증과 실행 상태

**[확인]** torch2.10 환경에서 CPU native1step을 포함한7개 fixture가 PASS했고 명령 종료코드0이었다. 실제 AdamW step1·SWA count2·finite crop loss, 작은K shape, native rolling와 wrapper parity, 미래 perturbation causality, 200시점 경계, 입력/weight 불변, 원형 save/load 및 wrapper checkpoint parity를 확인했다. `cpu_fixture_receipt.json`에 science fit0·backbone call0·smoke optimizer step1을 분리 기록했다. 공식 권장 torch1.8.1 환경 재현을 주장하지 않는다.

**[확인]** 추가 구조 fixture5개도 종료코드0으로PASS했다. TRAIN 경계 필터·빈 pool 거절, DataLoader tuple/value/RNG 관찰자 parity, 한도 직후 fetch 분리 시뮬레이션, mock benchmark loader의 pair 반환과 ID/checkpoint/E hash 거절을 확인했다. 이 구조 fixture의 actual optimizer step0·native encode0·TSFM call0이며, loader mock 검증은 실제 K5/K6 학습 checkpoint를 로드했다는 뜻이 아니다. 소스3개 compile도 통과했다.

이 문서는 방법 선정·구현 계약 문서다. 실제 CUDA4fits, fullVAL 예측오차, 전체 추론비용의 완료 상태는 `method_result.json`, benchmark 및 독립 검증 receipt를 따른다. 이 문서 작성 시 과학4fits/TSFM forward는 실행하지 않았다.
