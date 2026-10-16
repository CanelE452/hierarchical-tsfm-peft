# 원채널 과거 ridge 대조: 실제 학습 완료

[확인] Beijing·Jena·Electricity 전체 321채널·PeMS08 전체 170채널에서 `RAW_RIDGE`를 실제 CPU 적합했다. GPU/F 호출은 0이다. 아직 M_cal/V_select/T_final 예측·채점이나 배포 비용을 이 문서의 결과로 제시하지 않는다. 이후 잠긴 평가 실행기가 같은 cohort에서 수행한다.

## 정보와 학습 계약

각 채널 자신의 L512 과거를 32개 연속·비중첩 블록으로 나눈다. 각 블록 평균에서 해당 창의 마지막 값을 빼고 상수항을 더한 33개 특징으로 H개 미래의 현재값 대비 차이를 예측한다. 다른 채널의 값은 입력하지 않는다. 출력은 현재값 anchor와 예측 차이의 합이다. PeMS의 H는 12, 나머지는 48이다. 32블록은 고정된 시간 요약이며 원시 512개 위치를 모두 개별 계수로 학습하는 ridge는 아니다.

A_fit 50%의 실제 관측치로 mean/std를 적합하고 모든 합법 학습 origin을 stride1로 사용한다. alpha 후보는 사전 고정한 `.01,.1,1,10,100`이다. A_stop 10%에서 **H stride**의 관측 channel macro MSE가 최소인 alpha를 선택한다. 동일 점수면 더 작은 alpha를 택한다. 그 alpha를 고정하고 A_fit+A_stop 60%에서 mean/std를 새로 적합해 모든 합법 origin으로 다시 적합한다. M_cal/V_select/T_final 정답은 사용하지 않는다. 채널은 상수 여부와 관계없이 유지하고 std=0이면 1을 적용한다.

FP64 폐형식 목적식은 채널별 관측 전체 target 수 `N_c`를 사용한 다음 식이다. `beta`의 상수항은 정규화 penalty에서 제외한다.

```text
J = (1/C) sum_c [sum_(origin,h) observed*(prediction-target)^2 / N_c
                 + alpha*sum_(feature != intercept,h) beta[c,feature,h]^2]
```

각 채널·horizon의 실제 target 관측 mask로 normal equation을 계산한다. 결측을 0이라는 정답으로 학습하지 않는다. 같은 mask인 horizon은 같은 Gram을 재사용하되 RHS와 계수는 horizon별로 분리한다. intercept는 학습하고 penalty를 주지 않는다. BLAS는 2 thread로 제한했다. 모든 실제 적합에서 solver fallback은 0이었다.

이 방식의 학습 노출은 모든 합법 A origin이며 신경망의 0.25 epoch screen과 다르다. 계산량이나 최적화 예산이 동일한 비교라고 말하지 않는다. 다섯 alpha의 최하단 `.01`이 네 자료 모두 최상이었다. 따라서 이 grid에서 선택한 대조이며 전역 최적 alpha·충분한 탐색·수렴을 증명하지 않는다. 승인 없이 grid를 확장하지 않았다.

## 실제 완료 수치

```text
dataset       C    H   search origins  refit origins  A_stop origins  selected alpha  A_stop MSE   parameters
Beijing       11   48      9,960          12,063              43          .01          .6205753143    17,424
Jena          21   48      9,923          12,020              43          .01          .3592004568    33,264
Electricity  321   48     11,561          13,985              50          .01          .3249947439   508,464
PeMS08       170   12      8,405          10,190             148          .01          .1058799930    67,320
```

A_stop MSE는 alpha 선택에 사용한 A50 scaler상의 개발 점수다. A60 재적합 모델의 최종 평가 점수나 독립 일반화 검증으로 읽지 않는다. 파라미터 수는 `C*33*H`이며 scaler 통계나 저장 manifest를 학습 계수로 더하지 않았다.

```text
dataset       final fit end  search wall(s)  refit wall(s)  total wall(s)  CPU process(s)
Beijing          12,622          .42674          .49833          .94058          1.18750
Jena             12,579          .23660          .27880          .52814           .54688
Electricity      14,544         4.63639         5.54770        10.33695         10.87500
PeMS08           10,713         1.25122         1.44155         2.72619          2.85938
```

각 CPU 적합 프로세스 안에서 측정한 시간이다. GPU kernel time이나 장비를 독점한 속도가 아니다. 병행 GPU 학습의 CPU 준비와 경합할 수 있으므로 배포 지연은 별도로 같은 조건에서 측정한다.

## 과거 창 reset과 수치 검증

`PrefixFeatures`는 O(T*C) prefix sum과 다음 관측 위치를 유지하고, 각 채널의 N*33 특징만 생성한다. N*L*C 전체 과거 창을 쌓지 않는다. 전체 forward-fill을 그대로 쓰면 창 이전의 값이 유입되므로 창 시작부터 첫 관측까지의 기여를 빼서 Amean 제로로 되돌린다. 이는 `data_contract.history_window`의 창별 reset과 같은 의미다.

CPU synthetic test는 구현 전 3실패를 확인한 뒤 3통과했다. A_stop stride 수정 시험에서도 `21 != 6` 실패를 확인하고 H stride로 고쳐 통과했다. 긴 결측 구간을 포함한 원점에서 과거 특징을 1e-12까지 대조했고, 미래만 바꾸거나 비관측 target 값을 바꿔도 특징·계수·alpha가 달라지지 않는 것을 확인했다.

실제 네 자료의 저장 NPZ를 `allow_pickle=False`로 다시 읽었다. A 안의 고정 first/middle/last 각 3창에서 전체 채널의 `history_window` 대비 특징 최대 차이는 0이고, 별도의 명시적 dot product 대비 예측 최대 차이는 1.78e-15였다. 자료별 고정 3채널에서 모든 합법 A60 학습 원점의 masked normal equation을 다시 계산했으며 KKT 최대 잔차는 3.82e-17이었다. 이는 저장 계수의 산술·구현 검증이며 독립적인 별도 학습 재현을 뜻하지 않는다.

## 첫 선택 stride 불일치와 보존

처음 네 시도는 A_stop을 stride1로 평가했다. 공통 `fit_base.py`의 H stride 계약과 다르다는 것을 확인하고 최종 평가에 쓰기 전에 수정했다. 각 run의 `attempt_01_dense_astop/`에 이전 receipt와 `ATTEMPT_STATUS.json`을 보존하고 **INVALID_FOR_REGISTERED_PRIMARY_SELECTION**으로 명시했다. 이전 private NPZ도 같은 이름의 private attempt 폴더에 옮겨 보존했다. 이전 소스는 원 receipt와 일치하는 SHA256 `a9d8fd7739deb5c1f72e053b325a9a2e6d8b91dd8d7611c4e509a9db327d4777` 그대로 private에 보관했다. 이전 receipt의 원래 artifact 상대 경로가 어디로 이동했는지는 `ATTEMPT_STATUS.json`에 적었다.

수정 뒤 네 자료의 alpha 선택과 A60 재적합을 실제로 다시 실행했다. alpha는 모두 다시 `.01`이었고 이전 시도와 최종 시도의 계수가 전 요소 일치했다. 선택 원점 수와 선택 점수는 다르므로 이전 A_stop 점수를 등록된 주 결과로 가져오지 않는다. source/scaler/model fingerprint도 최종 계약에 맞췄다.

## 산출물과 호출 API

공개 산출물은 각 `runs/primary_<dataset>_E1_RAW_RIDGE_k0_s92601/`의 `RIDGE_FIT_RECEIPT.json`과 `RIDGE_REPLAY_RECEIPT.json`이다. private NPZ 상대 경로는 `ridge/<run_id>/artifact.npz`다. 계수 shape `[C,33,H]`, FP64, mean/std, block_edges, 선택 alpha, canonical_A60/actual_fit_end, source/data/feature/scaler/model hash와 Unicode contract JSON을 보존한다. pickled object나 원자료는 포함하지 않는다.

```python
artifact = ridge_baseline.load_artifact(private_npz)
features = ridge_baseline.temporal_features(x_normalized)       # [B,C,33]
predictions = ridge_baseline.predict_normalized(artifact, x_normalized)  # [B,H,C]
receipt, artifact = ridge_baseline.fit_task(task, cache_root, public_root)
```

입력은 저장 A60 mean/std를 사용해 창별로 채운 정규화된 과거이고 출력도 A60 정규화 좌표다. CPU 물리 입력부터의 전체 처리 비용에는 창 채움·정규화·33특징·ridge·역변환·CPU 반환을 포함한다. `model_sha`는 실제 계수·scaler·특징 fingerprint이고 `foundation_sha`는 `NO_FOUNDATION`이다. scaler fingerprint 계산식은 공통 `fit_base.scaler_fingerprint`와 같다.

다시 적합하려면 기존 receipt/artifact를 덮지 않고 별도의 private/public root를 전달한다. 환경변수에는 기존 자산 root와 새 private cache의 실제 로컬 경로를 설정한다.

```text
python -X utf8 research/informs_broad_validation_20261015_v1/test_ridge_baseline.py
python -X utf8 research/informs_broad_validation_20261015_v1/ridge_baseline.py --cache PRIVATE_CACHE --public PUBLIC_CAMPAIGN --datasets beijing jena electricity pems08
```
