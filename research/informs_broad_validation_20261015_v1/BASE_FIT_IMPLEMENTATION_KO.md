# 기본 모델 탐색·재적합 실행기

## API와 실행 범위

`fit_base.fit_base(spec, task, cache_root, public_root, foundation, deadline)`는 호출자가 이미 준비한 허용 데이터와 foundation을 사용한다. 데이터 다운로드, foundation 로드, subprocess 생성, Git 작업은 하지 않는다. `deadline`은 Unix wall-clock 종료 시각 또는 `None`이다.

`spec`의 주요 필드는 `run_id`, `family`, `latent_k`, `seed`, `epochs`, `lr`, `fold`, `microbatch`, `effective_batch`, `device`다. 기본 epoch 후보는 `[0.25, 1, 3]`, microbatch는 4, effective batch는 16이다. `DIRECT_G`, `DIRECT_B2`, `DIRECT_B4`는 `foundation=None`을 지원한다. 실제 F의 `row_microbatch=128`, `checkpoint_inputs=False`, eval/frozen 상태와 장비 배치는 호출자가 설정한다. 실행기 자체가 GPU를 선택해 모델을 내려받지 않는다.

공개 보조 API는 다음과 같다.

- `create_model(spec, task, fit_end, foundation) -> (model, normalization)`
- `predict_model(model, x, foundation, family, latent_k=None) -> prediction`
- `scaler_fingerprint(mean, std) -> sha256`

`create_model`은 지정 범위의 통계/PCA와 같은 seed의 새 모델을 생성한다. LoRA 모델을 직접 생성한 호출자는 반드시 `close()`로 hook과 원 F를 복구해야 한다. `fit_base`의 내부 단계는 정상 종료와 예외 경로에서 이를 수행한다.

## 고정한 비교 조건

1. Search는 fold별 `A_fit`의 모든 합법 origin을 stride1로 사용한다. 한 epoch는 이 전체 origin 집합의 순열 한 번이다. Fractional epoch는 `ceil(epoch × origin 수)` 노출이며, 다음 후보는 같은 순열 stream의 뒤를 이어 학습한다.
2. CP0를 포함해 전체 `A_stop` strideH origin에서 관측 target만 평가한다. FP64 SSE를 채널별 전체 관측수로 나눈 뒤 active channel macro 평균을 사용한다. Ordered multi-K는 각 K의 macro MSE를 동등 평균한다. 동점은 먼저 도달한 후보를 유지한다.
3. 선택한 epoch 수만 고정하고, `A_fit+A_stop` prefix에서 통계/PCA/seed 모델을 전부 새로 만든다. Search weights를 refit에 이어서 쓰지 않는다. 선택 epoch0은 새 CP0가 최종 부모이며 추가 최적화가 없다.
4. 독립 B0/B2/B4의 K=C는 E=identity, center=0으로 시작한다. Ordered multi-K의 Kmax=C는 공통 PCA prefix를 사용한다. `PCA_FIXED`/`PCA_FIXED_BOLT`는 중심화 PCA의 고정 선형 encoder/inverse이며, 학습 adapter의 현재값 anchor/lag7 decoder와 구분한다.
5. ADAPTS는 `networks.AdaPTSLinearPort`의 실제 deterministic affine encoder/decoder 포트를 사용한다. B0/B2/B4의 anchor/lag decoder로 바꾸지 않는다. DirectMixer와 temporal front는 해당 파일에 기록된 작은 캠페인 변형이며 원 논문 benchmark 재현으로 주장하지 않는다.
6. 최적화는 AdamW, weight decay0, clip norm1이다. `training.macro_loss`의 전체 train-origin 관측수 보정을 사용하므로 mini/microbatch별 관측수 차이로 채널 가중치를 바꾸지 않는다. F를 통한 학습에 `no_grad`를 걸지 않는다. `no_grad`는 A_stop 평가에만 사용한다.

Search의 FastWindows에는 A_stop 끝까지, refit에는 A_stop 끝까지의 prefix만 전달한다. M_cal/V_select/T_final 미래를 이 실행기가 window로 materialize하지 않는다. 정상화 metadata에는 `fit_end`와 `train_end`를 명시한다.

## 캐시·재개·영수증

Private 출력은 `runs/{run_id}/search/`, `refit/`, `selected_bundle.pt`에 저장한다. 각 checkpoint는 새 번호의 파일로 생성하여 과거 checkpoint를 보존한다. 공개 출력은 같은 run 상대경로 아래 SOURCE_CONTRACT, 단계별 event/receipt, flat CURVE.csv, BASE_FIT_RECEIPT.json이다. 공개 JSON에 실제 private cache 절대경로를 기록하지 않는다.

Source 파일의 실제 SHA256, spec와 L/H/C/채널 순서/fold/데이터 array fingerprint, data SHA, F model SHA를 묶는다. 단계별 평균/표준편차의 공통 `scaler_sha`와 PCA/center/fit_end를 포함한 `initialization_sha`도 별도 기록한다. 공통 scaler 알고리즘은 float32 mean/std 각각의 shape/dtype/raw-byte SHA를 `training.fingerprint`로 묶는 방식이다.

같은 run_id에서 source/config/data/model contract가 달라지면 실행을 거부한다. 완료 캐시는 선택 artifact의 실제 SHA까지 확인하고 재사용한다. Epoch schedule 연장은 새로운 실행 계획/ID 또는 상위 전용 확장 경로가 필요하다. `[.25]`를 `[.25,1,3]`으로 바꾼 동일 ID를 자동 재사용하지 않는다.

미완료 checkpoint는 모델·optimizer·sampler 순열 위치·Torch RNG·필요한 CUDA RNG를 보존한다. 다음 호출은 `training.restore_bundle`로 이를 복구한다. A_stop 평가 도중 종료 시각에 도달하면 부분 평가를 선택 점수로 쓰지 않고 BUDGET_LIMIT으로 저장한다. 종료 시각 검사는 update/평가 microbatch 경계에서 수행하므로 이미 시작한 GPU 연산 시간까지 포함한 강제 실시간 종료를 보장하지 않는다.

선정 artifact에는 model state, normalization/PCA metadata, spec, refit contract, search/refit summaries, model-state SHA가 들어간다. 공개 receipt는 실제 exposures/updates, 선택 CP, train origins, 고유 origin/target 시점, 관측 target 반복 노출, clipping, F logical/physical rows, 시간, process CUDA peak를 담는다. Peak는 해당 프로세스의 누적 측정값이며 독립 비용 측정이나 해당 경로의 고유 activation 사용량을 뜻하지 않는다.

`status=COMPLETE_VALID`는 요청한 탐색·재적합 artifact가 유효하게 완성되었다는 뜻이다. `[.25]` 실행은 `training_status=LIMITED_SCREEN`이며 1/3 epoch 미완료를 최소 커버리지 충족으로 표현하지 않는다. 3 epoch 마지막 CP가 선택되면 `UNDERTRAINED_OR_BUDGET_LIMIT`, 그보다 이른 CP도 수렴 검증 없이 `SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED`로 남긴다.

`last_cp_best`는 마지막 A_stop 점수가 전체 최저와 같은지, `last_scores_still_falling`은 마지막 두 평가 사이의 엄격한 하락 여부다. 제한된 화면에서도 이 둘을 별도로 보존한다. 두 점의 하락은 장기 수렴이나 개선 지속을 증명하지 않는다.

## CPU 합성 검증

`test_fit_base.py`의 최초 3 RED 실패 후 구현했다. 최종 5개 unittest와 10가족 subtest가 통과했다.

- 합성 DIRECT_G 실제 search/refit, 재적합 mean의 A60 범위, artifact 생성, 정확한 contract 재사용과 변경 LR 거부.
- K=C 독립 B0/B2/B4의 E와 center bitwise 초기값.
- optimizer/RNG/다음 origin batch 복구 후 다음 update의 모델 state bitwise 일치.
- deadline 중단 후 전체 fit 재개와 무중단 실행의 최종 model-state SHA 일치.
- B0/B2/B4/causal/bidirectional/ADAPTS/PCA_FIXED/LoRA/DIRECT_B2/DIRECT_B4 합성 CPU fit과 frozen F hash, LoRA hook 제거.

이 검사는 실제 데이터 성능, 실제 Bolt의 수렴, GPU peak 또는 배포 비용을 검증하지 않는다. 실제 GPU 실행과 별도 추론/비용 감사는 상위 캠페인 실행기의 책임이다.
