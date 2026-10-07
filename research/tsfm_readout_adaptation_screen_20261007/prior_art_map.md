# 선행 중복 지도 — 2026-10-07

[확인] 공식 원문/저자 공개본·코드를 아래 깊이로 읽었다. 세부 미확인은 검색 범위의 공백이며 논문에 없다는 판정이 아니다. S1은 기존 표준 대조를 현재 저장소의 고정 계약에서 보충하는 실험이다. 순서·새 이름·시계열 적용만으로 신규성을 주장하지 않는다.

## LogME — ICML2021

Problem/input: 사전학습 모델의 전이 성능 순위; frozen feature F와 target Y. Variables/gradient: Bayesian linear readout 가중치를 적분하고 prior/noise precision을 fixed-point 최적화; backbone gradient 없음. Decision cost: feature extraction+SVD+evidence 계산, 원문 비용도 특징 추출을 포함. Domain: 이미지 분류/dSprites 회귀/GLUE. Overlap: frozen-feature transferability 진단. Unverified difference: 좋은 readout 적합성이 같은 HEAD 부모 이후 LoRA의 추가 이득을 예측하는지는 확인하지 못했다; 결측/시간의존/forecast horizon 적용 별도.

열람: 초록, 본문§§3–5, Algorithm1, 회귀/비용 Table4; 코드·증명 전수 없음. [공식 원문](https://proceedings.mlr.press/v139/you21b/you21b.pdf).

## AutoPEFT — TACL2024

Problem/input: PEFT 종류/크기/삽입층의 성능–parameter-cost Pareto 탐색; 후보 configuration, TRAIN loss/DEV score. Variables/gradient: adapter/prefix 가중치 학습; SAAS-GP/NEHVI architecture search는 gradient 없이 수행. Decision cost: random100+BO100, 후보1epoch proxy; reported1.9%는 RTE search를 전체 GLUE에 분산한 특정 분모. Domain: GLUE/SuperGLUE, BERT/RoBERTa/T5. Overlap: 경로·비용 선택/짧은 후보 fit. Unverified difference: 같은 native HEAD 이후 추가 이득 사전판별은 미확인; standalone head-only를 읽은 baseline 목록에서 식별 못했지만 없다고 단정하지 않음.

열람: 원문§2 Algorithm1/수식, §§4–5/limitations; 코드 없음. [공식 원문](https://aclanthology.org/2024.tacl-1.29.pdf).

## LP-FT — ICLR2022

Problem/input: pretrained feature distortion과 ID/OOD 절충; penultimate features와 ID labels. Variables/gradient: LP는 linear head만, 이후 FT는 LP head 초기화에서 전체 모델 학습. Decision cost: LP+FT/LR selection; ImageNet은 두 단계를 각 절반epoch로 compute를 맞춤. Domain: 이미지 shift10개. Overlap: 헤드 먼저 내부 적응 순서와 좋은 초기화; LP comparator 존재. Unverified difference: pretrained nonlinear Bolt head/LoRA/같은 selected-head 부모의 HEAD_CONTINUE와 추가update 통제/사전 selector는 동일하지 않으며, 차이의 기여는 미검증.

열람: §§2–4, §4 recipe와 일부 추가 architectures/heuristics; 증명/코드 전수 없음. [저자 원문](https://arxiv.org/html/2202.10054v1).

## Time-PEFT — ICML2026

Problem/input: TSFM 적응 이득과 데이터 복잡성; spectral entropy/lagged transfer entropy. Variables/gradient: LoRA/frequency projection/shared down/channel-specific up/forecast head joint 학습, 기존 backbone 동결. Decision cost: complexity 계산+적응; exact PSD/TE estimator/window/discretization과 지표 wall-time 미확인. Domain: chaotic/ECG7+표준6자료, TSFM5개. Overlap: 언제 적응이 유리한가에 가까운 선행. HeadOnly 실제 존재, non-zero-shot arms는 head도 학습. Appendix D.1.2는 amazon/chronos-bolt-small, Appendix A는 original encoder-decoder HeadOnly/LoRA 대조도 명시. Unverified difference: exact output_patch_embedding 경계/q/loss/norm, 같은 selected-head 부모 continuation, 이후 이득 진단 계약은 미확인. “Bolt HeadOnly 선행 없음”을 주장할 수 없음.

열람: 공식 최종PDF pp1–9, Appendix A/B/D.1, 이론 일부; 공식코드10파일 검색/MOMENT allowlist/forward. complexity estimator는 scoped 코드에서 미확인. [공식 PDF](https://raw.githubusercontent.com/mlresearch/v306/acc326bcb01ce154c084938f42db80259d168431/assets/na26b/na26b.pdf), [고정 코드](https://github.com/kaist-dmlab/TimePEFT/blob/ea4e7e1887bb35587bab7ea93e2af3685ac55852/run.py).

## TRACE — 저자 arXiv v3

Problem/input: 큰 long-horizon head/LoRA module 선택; TRAIN labels와 validation gradients. Variables/gradient: reconstructed head/LoRA/gates joint fit; validation random masking 뒤 gate gradients 평균으로 pruning. Decision cost: 이미 적응하면서 validation backward M8 추가, Table6의 training/inference 비용; vanilla LoRA보다 training 시간이 긴 조건 존재. Domain: MOMENT forecasting/anomaly/NLP. Overlap: 내부 모듈 선택/head 재구성/head+LoRA. §5.2의 MOMENT_LP는 head 외 전부동결, LP+LoRA/reconstructed-head-only 비교도 존재. Unverified difference: native Bolt nonlinear head, 같은 selected-parent/update 대조, 내부 적응 전 추가이득 판별은 미확인. “based on linear probing”만으로 같은 checkpoint continuation이라 단정하지 않음.

열람: v3§§4–5/알고리즘/비용/resource 부록; publisher fulltext 접근 실패, Neurocomputing 최종판과 v3 동일성 미확인; 코드 없음. [열람 원문](https://arxiv.org/html/2503.16991v3).

## AdaPTS — ICML2025

Problem/input: frozen univariate FM의 multivariate probabilistic adaptation; multichannel history와 targets. Variables/gradient: enc/dec/stochastic adapter; backbone weights 고정이나 encoder에는 FM을 통과하는 input gradient. Decision cost: MOMENT linear-head를 먼저 fit/freeze(AppC.2), adapter/HPO/validation. Domain: 주요4자료/MOMENT-small, TTM/TimesFM 예비평가. Overlap: head-first 절차와 Fig7 decoder-only ablation. 공식 LinearDecoder는 Identity encoder와 bias 있는 Linear decoder; 원채널/RevIN 없는 경우 C→C affine OUTPUT와 가까움. Unverified difference: identity init/pinned Bolt q0.5/48/norm/selection은 재현조건 차이이며 자체 신규성 아님. OUTPUT는 전체 확률형 AdaPTS를 대표하지 않음.

열람: 최종PDF pp1–8/C.2, adapter fit/forecast/LinearDecoder 코드. [공식 PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/benechehab25a/benechehab25a.pdf), [고정 코드](https://github.com/abenechehab/AdaPTS/blob/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapters.py).

## Hyperband — JMLR2018

Problem/input: 후보 hyperparameters에 자원 배분; resource별 VAL loss. Variables/gradient: selector는 model gradients 안씀, 후보 runner가 지정 학습변수 fit. Decision cost: successive-halving brackets의 부분 fit/VAL; 한 실행 자원(s_max+1)B, epoch/iteration/data/features가 resource. Domain: CNN/kernel/다양한 선택 tasks. Overlap: 짧은 pilot 기반 경로 선택은 표준 원리. Unverified difference: HEAD/LoRA step비용·수렴/cache·공통 부모를 같은 total budget에서 계상하는 TSFM 계약과 pilot 대비 진단의 risk/총비용 우위는 별도 검증.

열람: §§1–3 Algorithm1, §§4.1–4.4/자원 주의; 증명/코드 전수 없음. [공식 원문](https://jmlr.org/papers/volume18/16-558/16-558.pdf).

## 현재 주장 결정

유지: 출력/헤드 표준 대조의 공정한 현재 저장소 검증과 같은 HEAD 이후 이득을 구분할 필요.

축소: “선행에서 head-only가 빠졌다”, “HEAD→LoRA 순서가 새롭다”, “시계열 적용 자체가 신규성”은 주장하지 않는다.

보류: 저비용 진단이 LogME/복잡성/같은 총예산 pilot보다 좋다는 주장, 내부 정보 부족/LoRA 필수성, 보편적 최적경로.

가장 강한 반론: 충분한 헤드 적응·표준 파일럿만으로 같은 결정을 더 싸게 할 수 있고, Time-PEFT의 실제 Bolt HeadOnly 대조와 차이도 exact source/cost 관점에서 작을 수 있다. S1만으로 이 반론을 해소할 수 없다.
