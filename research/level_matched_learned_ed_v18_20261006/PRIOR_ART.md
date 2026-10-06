# 선행 구조와 이번 matched E/D 대조

[확인] AdaPTS는 ICML 2025 논문이며 PMLR volume 267, pp. 3731–3748에 게재되었습니다. 저자는 Abdelhakim Benechehab, Vasilii Feofanov, Giuseppe Paolo, Albert Thomas, Maurizio Filippone, Balázs Kégl입니다. 공식 출판 페이지가 저자 저장소를 software로 연결합니다. [PMLR 출판 기록](https://proceedings.mlr.press/v267/benechehab25a.html), [공식 저장소](https://github.com/abenechehab/AdaPTS).

구조 감사는 공식 저장소의 commit `8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278`을 고정했습니다. commit 날짜는 2026-01-13이며 논문 출판 뒤의 공식 코드 revision입니다. 이를 ICML 실험 당시의 정확한 release나 독립 실행 재현이라고 부르지 않습니다. 다운로드한 정확한 UTF-8 raw bytes의 hash, 실제 1-based 행 번호, section hash와 immutable URL은 `source_binding.json`에 기록했습니다. 웹 도구의 추출 행 번호와 원파일 행 번호는 다를 수 있습니다.

## 공식 코드에서 확인한 사실

- [확인] `LinearAutoEncoder`의 encoder는 C→K, decoder는 K→C인 선형 layer입니다. 양쪽 constructor가 bias 인자를 생략하므로 기본 additive bias를 포함합니다. [공식 encoder/decoder, lines 385–391](https://github.com/abenechehab/AdaPTS/blob/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapters.py#L385-L391), [PyTorch Linear 기본 bias 정의](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html).
- [확인] Torch encoder/decoder 경로는 tensor를 직접 반환하여 adapter gradient 경로를 유지합니다. [Encoder, lines 514–520](https://github.com/abenechehab/AdaPTS/blob/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapters.py#L514-L520), [Decoder, lines 539–548](https://github.com/abenechehab/AdaPTS/blob/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapters.py#L539-L548).
- [확인] `adapter_supervised_fine_tuning`은 transform_torch → FM forecast → inverse_transform_torch 순서로 미래 예측을 구성합니다. FM learner를 eval로 두고 optimizer에는 adapter parameter만 넣습니다. [Forward, lines 480–504](https://github.com/abenechehab/AdaPTS/blob/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapts.py#L480-L504), [Optimizer, lines 506–510](https://github.com/abenechehab/AdaPTS/blob/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapts.py#L506-L510).
- [확인] 이 method 안에는 모든 FM weight를 명시적으로 `requires_grad_(False)`로 바꾸는 구문이 없습니다. 따라서 위 사실은 FM을 optimizer로 업데이트하지 않는 경로를 확인한 것입니다. eval mode 자체가 autograd 차단을 뜻하지 않으며, 공식 각 FM wrapper의 encoder까지 gradient 전달 성공을 여기서 실제 실행했다고 주장하지 않습니다.
- [확인] likelihood가 없는 deterministic branch의 decoded forecast loss는 MSE입니다. likelihood adapter에는 다른 확률적 loss branch가 있습니다. 공식 일반 MSE와 이번 프로젝트의 observed-mask channel-macro MSE reduction은 동일 정의로 간주하지 않습니다. [Prediction loss, lines 535–545](https://github.com/abenechehab/AdaPTS/blob/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapts.py#L535-L545).
- [확인] `coeff_reconstruction`의 기본값은 0.0이며 reconstruction 항은 양수일 때 더해집니다. 다른 adapter 종류의 KL 분기는 deterministic Linear E/D와 구분합니다. [Default, lines 373–389](https://github.com/abenechehab/AdaPTS/blob/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapts.py#L373-L389), [Optional auxiliary, lines 547–559](https://github.com/abenechehab/AdaPTS/blob/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapts.py#L547-L559).

## 이번 비교의 이름과 범위

공개 이름은 **MATCHED_LEARNED_ED** 또는 **AdaPTS-style matched Linear E/D control**입니다. 고정 K에서 learned E/D만 forecast loss로 학습하는 선택과 기존 fixed E/D+external residual G 선택을 직접 대조합니다. full AdaPTS probabilistic framework를 재현하지 않으며, 결과를 AdaPTS 전체와의 승패로 쓰지 않습니다.

요청된 arm은 bias=False, parent TRAIN PCA U transpose/U 초기화, parent pinned Chronos-Bolt-small, native q=0.5, L512/H48와 기존 관측-mask macro forecast MSE를 사용합니다. G·level persistence·RAW bypass·LoRA·reconstruction auxiliary·RevIN은 없습니다. 공식 기본 bias 및 일반 loss reduction·recipe와 다른 부분을 숨기지 않습니다. 세 자료의 TEST는 이전에 노출된 후속 개발 자료입니다.

```text
Attribute                MATCHED_LEARNED_ED                 ORIGINAL_LEVEL
E/D                      Learned, bias=False                 Fixed parent TRAIN PCA
Target-trained module    E and D                             Existing selected residual G
Latent rows              Same fixed K                        Same fixed K
Backbone weights         Frozen parent Bolt-small            Frozen parent Bolt-small
Training autograd        Retained through backbone to E      Existing residual-only route
Forecast output          Decode only                         Main decode + level + G
Auxiliary reconstruction None                                None in this new control
```

E/D 약100–250개와 G 17,920개는 예상 규모 설명이며, 실제 열거 값이 source of truth입니다. 이것은 parameter-matched contest가 아닙니다. Encoder 학습에 필요한 backbone-through gradient 비용은 이번 비교가 측정하려는 tradeoff입니다. 동일 세션 training-path probe는 convergence time을 대신하지 않으며 과거 LEVEL fit 시간과 새 ED 시간을 단순 비율로 비교하지 않습니다.

이번 source audit에서 모델 forward·fit·TEST prediction·새 metric은 수행하지 않았습니다. 양성·음성·혼재 결과와 실패/partial을 같은 방식으로 보존하고, 최종 주장 keep/shrink/hold는 봉인된 평가·비용 결과를 읽은 뒤 작성합니다. 자체 검산을 독립 재현·선행 부재·신규성 인증으로 확대하지 않습니다.
