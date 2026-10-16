# 과거 규약의 공개용 링크 보정 사본

원문 SHA256: `3100c3f435e3d60cee6633e2dec20aaaf9c80f7cb25d65e3e1a5b005534d4250`. 원문은 로컬에서 그대로 보존한다. 아래 내용은 하위 폴더에 복사되어 달라진 Markdown 링크 기준만 보정한 사본이며 최신 실행 규약은 상위 폴더의 문서를 따른다.

# 실제 배포 비용 측정 계약

[deployment.py](../deployment.py)의 `Route`는 저장 예측을 읽지 않고 실제 물리 입력과 원관측 mask에서 모델을 실행한다. [measure_costs.py](../measure_costs.py)는 실제 forward trial을 기록한다. 이 문서를 작성한 담당자는 CPU 합성 10개 검사만 실행했으며, 실제 GPU 비용 측정은 root의 독점 자원 큐가 수행해야 한다.

## 실제 배포 경로

```python
route = Route(route_spec, task, cache_root, model_path, device="cuda")
physical_prediction = route.predict_physical(physical_windows, observed_windows)
route.close()
```

입력 shape는 [B,L,C], 출력은 [B,H,C] 물리 단위 CPU 배열이다. shape·유한 관측값 검사, 창별 A평균 초기화/과거 방향 fill, CPU 정규화, 장치 전송, 실제 P/E/F/D/G/H, 역변환·동기화·CPU 반환이 whole 구간에 들어간다. 모든 관측이 유한하면 같은 의미의 fill 복사를 사용한다. 입력 창 밖의 이전 관측이나 미래값을 읽지 않는다.

`route_spec.family`는 RAW/CURRENT/SEASONAL 또는 fit_base 가족명(B0/B2/B4/DIRECT_G/DIRECT_B2/DIRECT_B4/LORA/ADAPTS/PCA_FIXED_BOLT/LAG_CAUSAL/LAG_BIDIR), COMBINATION, ALPHA다. 학습 모델은 `artifact_cache_relative_path`와 선택적으로 `artifact_sha256`를 받는다. private cache 밖 checkpoint는 거부한다. artifact의 task SHA·실제 parameter tensor SHA·평균/표준편차·원 F hash를 검산하며 `expected_prediction_model_sha`를 제공하면 calibration cache의 model SHA도 대조한다. 한 모델의 prefix K를 측정할 때 `latent_k`만 요청 K로 지정하고 실제로 그 K만 F에 전달한다.

RAW_RIDGE는 `ridge_baseline.py`가 생성한 NPZ를 로드하고 실제 native `predict_normalized`를 실행한다. 계수 [C,33,H]·데이터 hash를 검증하고 F를 생성하지 않는다. NumPy FP64 모델이므로 `--device cpu`로 측정한다. GPU FP32 모델과 같은 장치·dtype 비용이라고 주장하지 않는다.

COMBINATION은 E2 `selected.pt`, `variant`, 필요한 `g_route`/`b_route`를 받는다. 공통 정규화는 기본 A_fit+A_stop이며 `canonical_fit_end`로 분할에 맞는 등록 경계를 명시할 수 있다. checkpoint의 scaler SHA가 다르면 거부한다. 부모의 원 normalization이 다르면 원단위로 역변환한 뒤 공통 normalization으로 바꾸어 H에 넣는다. 같은 normalization이면 CPU 전처리를 공유한다.

conditional은 g와 b, past_control은 g, legacy_residual은 b만 로드·호출한다. 사용하지 않는 b가 spec에 남아 있어도 Hpast는 해당 F를 생성하지 않는다. residual은 새 M_cal R을 사용하고 fixed G를 로드하지 않는다. ALPHA fitted 경로는 g와 b를 모두 호출한다. alpha0/alpha1 참고선의 단독 비용은 각각 원 g/b Route로 측정한다. 학습된 계수가0이라는 이유로 F 비용을 지우지 않는다.

Route는 기본적으로 자기 lazy `FoundationPool`을 갖는다. 같은 H·장치의 general session은 pool을 명시적으로 공유할 수 있다. fresh G/direct/current/seasonal worker에서 F 인스턴스 생성수가0인지 실제 검사한다. 조합 부모는 필요한 F 하나를 공유한다. 등록된 조합 부모는 Raw/B0/B2/B4이고 서로 다른 LoRA를 같은 F에 동시에 병합하는 구성은 지원하지 않는다.

## LoRA와 Raw parity

LoRA는 실제 원 adapter state를 로드하고 unmerged-hook 예측과 merged 예측을 실제 동일 입력에서 비교한다. 허용치는 atol1e−5/rtol1e−4다. 병합한 q/v target의 실제 tensor를 CPU에 보존하여, 반복되는 method 전환에서도 같은 merged tensor를 복사한다. 원 target과 adapter 보조 사본은 CPU로 옮겨 steady-state GPU memory에 비활성 GPU adapter 사본을 남기지 않는다. 단일 fresh worker에서는 timed trial 동안 병합 상태를 유지한다. general session은 다른 방법의 F 오염을 피하도록 방법 전환의 activate/deactivate를 타이머 밖에서 수행한다. 이 전환 비용을 steady-state 추론시간에 합치지 않는다.

`close()`는 원 target tensor·gradient flag를 복원하며 원 F 전체 hash를 확인한다. 병합 전후 검사·복구 검사는 추가 검증 비용이고 timed trial과 구분한다.

Raw는 row microbatch full/32/8/4를 동일 물리 입력에서 실제 재추론해 비교한다. 전체 성공을 가정하지 않는다. CUDA OOM은 그 row 설정을 BLOCKED_CUDA_OOM으로 남기고 가능한 설정을 계속한다. 첫 성공 설정이 parity reference이고 성공한 설정 사이 불일치가 있으면 비용을 유효 결과로 판정하지 않는다. microbatch별 실제 시간은 해당 row 설정을 지정한 별도 worker로 측정해야 한다. K/C를 속도비로 대신 사용하지 않는다.

## whole과 stage의 측정 범위

`measure_balanced(routes, physical, observed, warmups=10, timed_trials=20, stage_trials=20, lease=owner)`는 방법 시작 위치를 매 round 순환한다. warmup/timed/stage_probe 행을 분리한다. 독립 general-session 비교와 route별 fresh-process memory를 다른 scope로 남긴다. route가 하나인 fresh worker의 order rotation은 자명한 단일 방법 순서다. 여러 방법의 balanced 일반 세션은 별도 API 실행이 필요하다.

whole timed trial에는 stage 내부 동기화가 없다. stage_probe는 CPU_preprocess/host_to_device/P_E/F/D/G_or_direct/H_or_alpha/inverse_and_CPU를 CUDA 동기화 경계로 측정하므로 intrusive한 별도 scope다. probe의 whole 시간을 비계측 whole 시간과 혼합하지 않으며 stage 합계를 whole과 같다고 주장하지 않는다. 임의 저장 예측 접근을 F 비용으로 세지 않는다.

memory에는 resident 모델 parameter/buffer, active non-foundation GPU parameter/buffer, 실제 peak_allocated/reserved를 MiB로 기록한다. peak−resident를 순수 activation이라고 부르지 않는다. LoRA 병합 준비 및 parity 후 unused cache를 비우고 warmup 뒤 peak 통계를 초기화한다. 학습 peak는 이 파일의 추론 peak에 합치지 않으며 원 search/refit 학습 영수증을 별도로 사용한다. 호출자가 이미 F를 상주시킨 general G memory와 F가 전혀 없는 fresh G memory를 동일한 조건으로 묶지 않는다.

## GPU worker 실행

GPU worker는 실행 중인 root `GPUReservation`의 자손이어야 하고 root의 `child_environment()`를 전달받아야 한다. lease의 실제 OS lock·살아 있는 owner PID/create_time·서명된 왕복 handshake를 model allocation 전에 확인하며 각 parity/trial 전에도 검사한다. token·owner 인증값은 공개 영수증에 쓰지 않는다. 직접 GPU 명령을 무승인 독립 실행하도록 제공하는 것이 아니다.

```text
python research/informs_broad_validation_20261015_v1/measure_costs.py --worker --route-spec <private-route.json> --cache-root <private-cache> --model <pinned-local-Bolt> --batch 1 --warmups 10 --timed-trials 20 --stage-trials 20 --output <cost-receipt.json>
```

batch8은 별도 fresh process에서 같은 명령의 `--batch 8`로 실행한다. 기본 cost cohort는 등록된 V의 첫 strideH histories이며 V origin 수가 B보다 적으면 비용 배치 구성에 한해 반복하고 그 사실을 기록한다. 이는 평가 표본 확대가 아니다. 결과는 raw 물리 예측 SHA·실제 F stream/call·trial 순서·시간·memory·모델 로딩 시간·source/artifact hash가 포함된 JSON과 gzip flat trial CSV다. 실제 예측값·정답 배열·개인 절대경로는 게시하지 않는다. gzip 파일20MiB 상한을 검사한다.

일반 balanced session은 같은 명령의 `--worker` 대신 `--balanced`를 사용하고 `--route-spec`에 같은 dataset의 route dictionary 목록 JSON을 넣는다. cost condition ID는 Raw row variant를 포함해 서로 달라야 한다. 한 F pool을 공유하고 warmup10/timed20/stage20을 각각 순환하며 모든 method를 실제로 실행한다. GPU 실행은 이 모드도 살아 있는 owner의 child lease를 요구한다. 일반 세션의 공존 memory를 fresh worker의 단독 memory로 바꾸어 쓰지 않는다.

모델 로딩시간은 별도 필드다. warmup10/timed20/stage20을 충족하지 못하면 SMOKE_OR_PARTIAL_COST_PROTOCOL이다. 최소 trial을 채웠다는 것만으로 별도 학습·성능·일반화가 확인되지는 않는다. deadline·OOM·누락 자산으로 못 한 조건은 root registry/재개 큐에서 별도 미실행 상태로 유지해야 한다.

## 실제 CPU 검사

```text
python research/informs_broad_validation_20261015_v1/test_costs.py
```

모듈 부재 RED 후 CPU10 tests GREEN을 확인했다. 실제 toy forward를 통해 physical fill/reset·Current F0·학습 형식 G artifact F0·Raw stream count와 row parity·balanced rotation·GPU child 인증 거부·실제 q/v-LoRA merge parity와 반복 병합/복구 hash·prefix K의 실제 F stream 감소·Hpast의 불필요 Raw parent 로드 차단·native FP64 ridge의 실제 계수 forward와 F0·worker/balanced driver의 실제 toy JSON/gzip trial 생성을 확인했다. toy fixture는 성능 학습 결과가 아니며 GPU 지연시간 수치를 추정하지 않는다.

## 배포 경로 정밀도와 추가 경로

실제 비용 실행 전에 큰 물리 offset을 가진 합성 회귀 검사에서 FP32 역정규화가 작은 예측 차이를 양자화하는 문제를 확인했다. 허용오차를 넓히지 않고 **모든 경로의 역정규화와 CPU 물리 출력만 FP64**로 변경했다. F·어댑터·G의 내부 계산은 FP32이며, native ridge는 NumPy FP64 CPU 계산이다. 비용 영수증의 `dtype`는 내부 모델 dtype, `inverse_transform_dtype`와 `physical_output_dtype`는 FP64를 나타낸다. 전체 파이프라인 시간에는 이 변환과 FP64 출력 전송도 포함한다. 이 수선은 정규화 좌표에 저장된 기존 학습 예측 점수를 변경하지 않는다. 합성 동일 모델의 normalized replay 최대 차이가 0.0051937103에서 0으로 줄어든 것은 배포 경로 검산이며 연구 성능 개선이 아니다.

M-only Raw 채널별 bias NPZ 경로와 nested PCA 전체 basis NPZ의 실제 K-prefix 경로를 추가했다. 최대 K 전용 모델의 사후 절단은 artifact를 바꾸지 않고 생성 시 허용 inference grid만 확장한다. 원 학습 `k_grid=[C]`는 보존되며 작은 K에서 학습한 모델이라고 쓰지 않는다. 추가 CPU 검사에서 실제 부모 호출·K행 수·동일 F hash를 확인했다.

`prediction_cache.py` 및 `ordered_pilot.py`가 저장하는 학습 경로 예측 manifest의 `cost_record.F_calls`는 `batch × K × F_calls` 검산에 쓰는 논리적인 backbone 경로 배수다. 이는 여러 row chunk로 나누어진 PyTorch 호출 수와 다르다. `route_predictions.py`의 배포 재추론 manifest와 실제 배포 timer의 `F_calls`는 runtime에서 센 물리 forward 호출 수이며 `F_logical_streams`/`F_physical_streams`도 따로 기록한다. 저장 부모를 사용하는 결합 수치 계산의 F 호출은 0이다. 이를 실제 배포의 F 비용 0이라고 바꾸어 쓰지 않는다.

현재 loader의 cold setup 시간은 artifact·자료 계약 검산과 A60 통계/PCA 초기화 재구성도 포함한다. 이를 최소한으로 최적화된 제품 cold start라고 부르지 않는다. 정규화/P/E/F/D/역변환을 포함한 반복 추론은 setup과 분리해 직접 측정한다.
