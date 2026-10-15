# 실제 예측 학습과 시간 범위 재검토

**[확인] 예측 목적의 인코더 학습을 하지 않은 상태가 아니다. 실제 미래 Y에 대한 손실을 frozen Bolt의 입력까지 역전파한 E/D 공동 학습, 재구성 D 초기화, D 고정·E만 갱신하는 학습까지 실행했다. 다만 최신 비교들은 같은 FIT256/TUNE64의 반복이며, Robin의 고유 학습 기간은 짧다. 이것만으로 방법 전체의 불가능성이나 최적화 문제가 모두 배제됐다고 결론 내릴 수 없다.**

이 문서는 source와 JSON/MD metadata를 읽은 별도 검토다. 원자료·NPZ·모델을 열거나 학습·추론·비용 측정을 새로 실행하지 않았다. 아래의 후속 제안은 **미실행**이며 이번 게시 요청에 새 실험을 추가하지 않는다.

## 실제 실행한 목적과 초기화

[선형 E/D 실행 결과](runs/linear_e_train_v1/RESULT.json)는 COMPLETE이며, Robin C17/K4와 Beijing C11/K2 × L0/L7 × E 고정/갱신 × minibatch 순서 seed 2개의 **16 fit, 9,600 update**를 기록한다. [runner의 train_condition](run_linear_e_adaptation_v1.py)과 [LinearLagAdapter](linear_e_adaptation_v1.py)를 함께 확인했다.

- 배포식은 과거만으로 Z=E(X), 미래 latent=F(Z), 원채널 예측=current-C + lag-feature·D다. F의 parameter는 동결하며 optimizer에서 제외하지만 E까지의 입력 gradient는 유지한다.
- 손실은 실제 관측된 FIT 미래 Y의 채널 균등 정규화 MSE다. 전체 FIT 채널 관측수로 minibatch SSE를 가중한다. 재구성 손실만 학습한 비교가 아니다.
- fit마다 AdamW lr .001/weight decay 0/clip 1, batch 4, 600 update다. 0/100/200/400/600 checkpoint 중 TUNE MSE로 선택했다. 600 update는 2,400번의 window 추출, FIT256 기준 약 9.375번의 표본 추출량에 해당하며, 복원추출이므로 완전한 9.375 epoch라고 부르지 않는다.
- 초기 E는 FIT PCA, 초기 D는 같은 lag의 FIT FORECAST-LS다. 두 seed는 **동일 초기값에서 minibatch 순서만** 바꾼다. 독립 초기화 두 개의 검증으로 세면 안 된다.

[RECON 초기화 실행](runs/recon_init_sensitivity_v1/RESULT.json)은 Robin L7 두 fit·1,200 update COMPLETE다. [run_recon_init_sensitivity_v1.py](run_recon_init_sensitivity_v1.py)는 기존 FIT 과거로 적합한 PCA E와 L7_RECON D에서 출발해 **동일 actual-Y 예측 목적**으로 E/D를 함께 갱신한다. `joint_recon`이라는 후속 provenance 이름은 이 실험을 가리키며, 재구성 목적만의 joint training이라는 뜻이 아니다.

[고정 RECON D·E-only 실행](runs/fixed_recon_d_e_adaptation_v1/RESULT.json)도 두 fit·1,200 update COMPLETE다. [run_fixed_recon_d_e_adaptation_v1.py](run_fixed_recon_d_e_adaptation_v1.py)는 같은 RECON 초기값에서 D를 동결하고 E만 actual-Y로 학습한다. 공동 학습과 clipping 대상이 달라지는 procedure 비교이며 단일 인과 효과로 과장하지 않는다.

더 이전에도 [small-K 연구](../../smallk_predictive_alignment_20261013/README.md)에서 actual-Y 목적, teacher 초기화, anchor, pointwise/temporal encoder, 미래 정렬을 실행했다. [joint objective runner](../../smallk_failure_followup_20261014/run_joint_objectives_v2.py)는 이미 TUNE로 선택한 MAIN checkpoint를 같은 초기값으로 삼아 Y/Raw teacher/혼합을 추가 600 update 비교했다. 따라서 이를 무작위 초기화에서의 독립 재학습이나 새로운 미래 정보 획득으로 부르면 안 된다.

## FIT256의 고유 시간 범위

[검산된 cross-moment RESULT](runs/predictive_cross_moment_v1/RESULT.json)의 `datasets.*.timestamp_occurrences`를 읽었다. 아래 index는 실제 timestamp에 대응하는 정수 위치이며 시간 독립 표본 수가 아니다.

```text
자료       FIT 과거 slot / 고유 시점 / 범위 / 최대 중복
Robin      131072 / 1545  / 528–2072   / 127
Beijing    131072 / 20990 / 0–20989    / 7

자료       FIT 미래 slot / 고유 시점 / 범위 / 최대 중복
Robin      12288 / 1081  / 1040–2120  / 12
Beijing    12288 / 12288 / 512–21037  / 1

자료       TUNE 미래 고유 시점 / 범위 / 최대 중복
Robin      399  / 2121–2519   / 9
Beijing    3072 / 21038–28050 / 1
```

**[판단] Robin에서 같은 window 수나 update 수를 늘리는 것만으로 새로운 시계열 구간이 늘어나지는 않는다. 반대로 Beijing의 FIT 미래는 이미 slot마다 고유 시점이 달라, 두 자료의 실패를 모두 “심한 window 중복 때문”이라고 설명할 수 없다.** Beijing의 complete-HC FIT origin은 83/256이고 complete-C 미래는 10,881/12,288이다. 83을 채널별 마스킹 학습 전체의 표본 수로 대신하지 않는다.

[prepare_inputs.py](../../smallk_failure_followup_20261014/prepare_inputs.py)의 Robin 복원은 기존 `load_data(..., sizes=(256,64,None))`로 이어진다. [common.load_data](../../past_only_repr_diagnosis_20261012/common.py)는 원 TRAIN의 앞 80%를 FIT, 뒤 20%를 TUNE로 두고 합법 origin을 균등 선택한다. 최신 runner는 이 과거 전용 입력·actual-Y·mask·origin·FIT mean/std가 봉인된 payload를 그대로 사용한다. fill은 origin별 512 window 시작에서 FIT mean으로 초기화하는 causal forward fill이다. global series fill로 임의 교체하면 기존 비교 조건이 달라진다.

## 시간순 진단의 범위와 반례

[purged readout 결과](runs/purged_readout_stability_v1/RESULT.json)는 `fixed_full_FIT_upstream=True`다. [runner](run_purged_readout_stability_v1.py)는 원 FIT256을 시간순 A/B/C로 나누고 응답 범위가 겹치지 않게 purge한 뒤 **추가 STATIC readout만 A에서 fit, B에서 선택, C에서 평가**한다. E·원 D·전처리는 full-FIT에서 정한 기존 상태다. 그러므로 이것은 **A-only upstream representation 학습이나 전체 pipeline의 새로운 시간순 holdout 검증이 아니다.**

그렇다고 역사 전체에서 시간순 upstream 적합을 한 번도 하지 않았다고 쓰는 것도 틀리다. [Oct11 데이터 계약](../../level_compressor_fullscreen_20261011/DATA_CONTRACT.json)과 [data_contract.py](../../level_compressor_fullscreen_20261011/data_contract.py)는 fit/calibration/selection을 분리하고 normalizer·graph를 fit 구간에서만 적합한다. Robin canonical fit은 [528,1922), backtest fit은 [528,1503)와 [528,1713)다. Jena canonical fit은 [0,18344)다. [train.py](../../level_compressor_fullscreen_20261011/train.py)는 최대512개 TRAIN-fit origin pool에서 batch32로 2,880 update를 실행하며 frozen F를 생성하지 않는다.

[당시 보고서](../../level_compressor_fullscreen_20261011/REPORT_KO.md)에 실제 PointMLP/causalTCN 다섯 목적40 fit, 공식 TS2Vec small/large 8 fit, GTConvAE 계열12 fit이 명시돼 있다. [models.losses](../../level_compressor_fullscreen_20261011/models.py)는 reconstruction, variance/covariance guard, detached auxiliary control, latent self-forecast 목적을 구분한다. TS2Vec의 hierarchical contrastive loss는 자기과거 잠재 미래 예측 손실과 같은 목적이 아니다. 이 과거 실험의 시간 분할·정규화·손실·K와 최신 full-gradient 실험이 다르므로, 값을 같은 조건의 우열로 섞지 않는다. 당시 보호 접근 incident는 원 보고서대로 보존한다.

## 남는 비중복 질문

**[판단·미실행] 더 복잡한 decoder를 추가하기 전에, “같은 단순 E/F/D를 충분히 다른 TRAIN 시점에서 학습하고 모든 upstream 통계를 앞 구간에서만 결정했을 때도 이득이 뒤 구간으로 옮겨 가는가”를 확인하는 것이 현재 남는 작은 실질적 변경이다.** 새로운 구조보다 학습 구간의 유효 폭과 순서를 바꾸는 질문이다. Robin의 원 보호 전 TRAIN 자체가 짧으므로 FIT256→512만으로 해결한다고 약속하지 않는다. Beijing은 이미 넓은 미래 timestamp를 갖고 있어 이 변경의 반례 자료로 유지해야 한다.

이후 실제 실행하려면 먼저 안전 TRAIN 자산의 존재·경계를 확인하고 A-only scaler/center/E/D 초기 적합 및 TUNE barrier를 새로 봉인해야 한다. 기존 full-FIT E/D를 재사용하면서 A-only 검증으로 부를 수 없고, 새 scaler를 쓰면 Raw baseline도 같은 통계로 재계산해야 한다. 같은 source의 원 TRAIN을 더 촘촘히 추출하는 것과 더 넓은 고유 기간을 얻는 것은 별도로 기록해야 한다. 추가 연구는 이번 문서에서 실행하지 않았다.

## 확인한 결과 파일의 SHA256

```text
linear_e_train_v1             3f0cbd9176f2e7882778452e537564aab7445e1b88c1534fd48b1bce7ef639a4
recon_init_sensitivity_v1     f0fc3f95f1e1a61eff4ffc99b7bb7d62affd34afa064f6e29113bb0c37f52ec8
fixed_recon_d_e_adaptation_v1 4b1559216ea99937ca8e7e0f6cdac9885abc29479a6050d82336dcfa0305bb2b
predictive_cross_moment_v1    99bffd5831143ca35947099fef9567f89dd209d83451661decee0682b2b8e13c
purged_readout_stability_v1   4ad5ee73f789a3b21eab8d0b732b04bde2622e5380e915a07f226869a2a6becf
```
