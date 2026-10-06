# Matched learned E/D — fairness and execution boundary

[확인] 실행 상태는 BLOCKED_STOP_RESOURCE다. 아래는 새 학습 전 실제 loaded enumeration과 parent 계약이다. 새 matched training/inference 비용은 unavailable이며 과거 receipt를 새 cost probe로 표시하지 않는다.

```text
Shared contract
model_family_revision: amazon/chronos-bolt-small @ 772f3d25d38aec6d914c8949dab4462e2d46f5d8 (literal parent checkpoint)
context_horizon: L=512, H=48; FP32; TF32 off
point_forecast: native inverse-scaled q=0.5, first 48 steps
input: original C channels; B*K independent latent TSFM rows for both arms
preprocessing: literal TRAIN-only normalization, imputation, channel order, origins and observed-target mask
exposure: all three TEST groups were exposed before this follow-up; no independent confirmation
new_fit_count: 0; new_TEST_model_predictions: 0; optimizer_updates: 0
```

```text
Robin: C=17, K=5
field                         MATCHED_LEARNED_ED(step0)      ORIGINAL_LEVEL(reused)
backbone_parameter_count      47,718,016                   47,718,016
total_parameter_count         47,718,186                   47,736,106
trainable_component_count     170                          17,920 G (historically fitted)
parameter_bytes               190,872,744                  190,944,424
buffer_bytes                  376                          36
total_tensor_bytes            190,873,120                  190,944,460
E/D                           bias=False, trainable          fixed literal PCA
initialization                E=U.T; D=U                    TRAIN PCA U
G / level persistence         absent / absent                fitted G / included
training gradient path        through frozen backbone to E  external residual G only
new matched cost / resident   unavailable, gate prevented    unavailable, gate prevented
```

[확인] ED source: preflight.json `/units/robin/model_receipt`; PCA SHA256 `352532dd5a659db657de52c3333a7101c11345f82c499c1ec04d75256390e0f9`. LEVEL count source: `research/level_bolt_backbone_scaling_v1_20261006/preflight_checks.json` `/baselines/3/receipt`와 두 번째 seed receipt. LEVEL 값은 이전 load 기록이며 이번 회차의 cost 측정이 아니다.

```text
Peacock Education: C=13, K=4
field                         MATCHED_LEARNED_ED(step0)      ORIGINAL_LEVEL(reused)
backbone_parameter_count      47,718,016                   47,718,016
total_parameter_count         47,718,120                   47,736,040
trainable_component_count     104                          17,920 G (historically fitted)
parameter_bytes               190,872,480                  190,944,160
buffer_bytes                  244                          244
total_tensor_bytes            190,872,724                  190,944,404
E/D                           bias=False, trainable          fixed literal PCA
initialization                E=U.T; D=U                    TRAIN PCA U
G / level persistence         absent / absent                fitted G / included
training gradient path        through frozen backbone to E  external residual G only
new matched cost / resident   unavailable, gate prevented    unavailable, gate prevented
```

[확인] ED source: preflight.json `/units/peacock_education/model_receipt`; PCA SHA256 `2263f0d6a05433c1f840c28a60bd7e589dd257caff7e9b445c8a87704133e26a`. LEVEL count source: `research/level_bolt_backbone_scaling_v1_20261006/preflight_checks.json` `/baselines/5/receipt`와 두 번째 seed receipt. LEVEL 값은 이전 load 기록이며 이번 회차의 cost 측정이 아니다.

```text
Jena: C=21, K=6
field                         MATCHED_LEARNED_ED(step0)      ORIGINAL_LEVEL(reused)
backbone_parameter_count      47,718,016                   47,718,016
total_parameter_count         47,718,268                   47,736,188
trainable_component_count     252                          17,920 G (historically fitted)
parameter_bytes               190,873,072                  190,944,752
buffer_bytes                  540                          36
total_tensor_bytes            190,873,612                  190,944,788
E/D                           bias=False, trainable          fixed literal PCA
initialization                E=U.T; D=U                    TRAIN PCA U
G / level persistence         absent / absent                fitted G / included
training gradient path        through frozen backbone to E  external residual G only
new matched cost / resident   unavailable, gate prevented    unavailable, gate prevented
```

[확인] ED source: preflight.json `/units/jena/model_receipt`; PCA SHA256 `de502c98819a1757c5b92bf513a44f47c609275f66363feef8790009246ac283`. LEVEL count source: `research/level_bolt_backbone_scaling_v1_20261006/preflight_checks.json` `/baselines/7/receipt`와 두 번째 seed receipt. LEVEL 값은 이전 load 기록이며 이번 회차의 cost 측정이 아니다.

[확인] 실제 ED enumeration은 170 / 104 / 252다. LEVEL G 17,920과 parameter matched가 아니다. 전체 배포 가중치에는 동일한 47,718,016 backbone parameters가 포함되므로 작은 adapter count만으로 memory/latency 우위를 말할 수 없다.

[확인] 공식 LinearAutoEncoder는 bias 기본값 True를 사용한다. 본 bias=False, literal PCA 초기화, observed-mask channel-macro forecast MSE만 쓰는 arm은 full AdaPTS reproduction이 아니다. 구조 선택을 좁혀 묻는 control이다. [공식 source pin](https://github.com/abenechehab/AdaPTS/blob/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapters.py#L348-L395), [binding](source_binding.json).

Source SHA256: preflight.json `0539e35f58a14de2babe1525271d5b54ff61fd9fa761b5742200ec1ed1fc9cdd`; historical LEVEL enumeration `49d1191d38bb614715f6ee4e20c3d1ac35b7b11f5296bf32c3fa4f3dc4713766`.
