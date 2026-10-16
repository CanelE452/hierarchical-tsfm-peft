# E2 결합 구현과 검증 범위

[combination.py](combination.py)는 고정된 부모 예측의 시간 외 calibration을 구현한다. 이 담당자가 실행한 결합 검사는 CPU 합성 14개이며 실제 자료의 H·residual 학습이나 F 추론은 실행하지 않았다. 추가로 새 캠페인 전체 CPU 회귀 검사 114개와 subtest 26개가 통과했다. 실자료 실행 상태·수치는 root 큐의 실행 영수증과 결과표를 확인해야 한다.

## 호출 계약

```python
receipt = fit_combination(
    spec, task, parent_m, parent_v, g_m, g_v,
    mean, std, cache_root, public_root, deadline,
)
```

`task`는 `data_contract.load_task` 결과다. 공통 `mean/std`는 A_fit+A_stop 재적합 통계이며 scaler fingerprint는 `training.fingerprint`에 float32 mean/std의 `evaluation.array_fingerprint`를 넣어 계산한다. PCA/표현 초기화 hash와 별개다.

네 부모 dictionary의 필수 키는 `pred_normalized`, `target_normalized`, `observed`, `origins`, `channel_ids`, `metadata`다. 예측·정답·mask의 축은 [origin,H,C], mask는 bool, channel ID는 원 C의 같은 순서다. metadata에는 `data_sha256`, `scaler_sha`, `model_sha`, `cohort`, `split`, `parent_fit_end`가 있어야 한다. M/V는 같은 model SHA를 사용해야 하며 `parent_fit_end <= M_cal.start`여야 한다. 실제 task에서 정답·원 mask를 다시 가져와 정확히 대조한다. 달력/shape가 맞아 보여도 다른 정답·mask·채널 순서·normalization·cohort를 거부한다.

M_cal은 학습용 모든 합법 origin stride1의 **dense 부모 예측**이 필요하다. 평가 strideH 부모를 몰래 반복해 dense 학습으로 세지 않는다. V_select는 정확히 등록된 strideH cohort다. stage별 창 캐시는 M 끝 또는 실제 V 예측 마지막 target 끝까지만 materialize하여 T_final 값을 새로 정규화하거나 모델에 제공하지 않는다.

`spec` 필수 값은 `run_id`, `variant`다. 학습 기본값은 seed92601, learning_rate0.001, effective_batch16, microbatch4, epoch_endpoints[0.25,1,3], devicecuda다. 부모를 직접 호출하는 함수나 F 객체를 받지 않는다. 실제 실자료 실행과 부모 예측 생성은 별도 실행기가 담당한다.

## 실행하는 네 가족

- `alpha_family`: M_cal의 관측 채널 macro MSE 가중치로 FP64 전역 alpha를 닫힌형으로 적합한다. `g + alpha*(b-g)`의 alpha0/0.5/1/M-fitted 네 경로를 모두 저장한다. alpha는 [0,1]로 clip하며 분모0이면0이다. 저장된 배포 예측은 FP32이고 공개 점수·SSE/SAE/count는 그 저장 예측을 기준으로 계산한다. FP64 적합 목적값과 저장 FP32 예측 점수는 각각 남긴다.
- `conditional`: `g + H(past-current, g-current, b-g)`. channel-shared hidden32이며 출력층만0 초기화한다. 초기 예측은 정확히 g이고, 학습 후 비악화를 보장하지 않는다. G와 b는 detach된 고정 부모다.
- `past_control`: 같은 H 클래스·입력 폭·활성 파라미터 수를 사용한다. 마지막 `b-g` 위치를 실제 과거의 최근 H개 current-centered 값으로 바꾼다. 0 padding으로 이름만 맞춘 대조가 아니다. 이 경로의 배포에는 고정 b 예측이 필요하지 않는다. 실제 비용 실행기는 불필요한 b/F를 로드하거나 호출하지 않아야 한다.
- `legacy_residual`: 새 DirectG의 혁신량 R을 M_cal에서 actual Y−b 목적에 적합해 `b + R(past)`를 만든다. 구현은 `b + (new_DirectG(past)-current)`이며 CP0는 정확히 b다. A에서 적합한 고정 G의 예측을 무학습으로 더하는 경로가 아니다. 고정 G와 독립적인 새 파라미터이며, 이 배포 경로에서 고정 G를 호출할 필요는 없다.

Raw/B0/B2/B4-conditioned는 부모 cache만 바꾸고 같은 H 학습 계약을 쓴다. Hpast는 b를 입력에 사용하지 않으므로 다른 b에 대해 여러 번 수행해도 독립적인 방법으로 세지 않는다. root 실행기는 동계약이면 재사용 사실을 명시할 수 있다.

전역 예측 결합은 오래된 표준 대조이며 새로운 기여로 주장하지 않는다. [DeepGLO 원문 §5.2](https://proceedings.neurips.cc/paper_files/paper/2019/file/3a0844cee4fcf57de0c71e9ad3035478-Paper.pdf)는 공통 예측을 개별 예측기의 입력으로 사용하는 원리를 제시한다. 이 구현은 고정 Bolt·기존 G·소형 정적 H와 별도 M/V 계약을 사용하며 DeepGLO 원모델이나 학습 결과의 재현이 아니다.

## 학습·재개·산출물

실제 M_cal 전체 origin을 seed 고정 permutation으로 노출한다. global M 관측수로 채널 가중치를 정하고 effective batch 안의 microbatch loss를 같은 분모로 누적하여 실제 effective16 목적과 맞춘다. AdamW, weight_decay0, gradient clip1이다. CP0와0.25/1/3 epoch 끝의 V macro MSE로 checkpoint를 선택한다. 마지막 checkpoint가 best이면 `UNDERTRAINED_LAST_CHECKPOINT_BEST`이고, 이외 선별 종료도 수렴을 증명하지 않는다.

원 부모 cache array/metadata의 전후 fingerprint를 대조한다. 부모 weight 재계산·변경은 하지 않으며, model SHA는 입력 영수증에서 결속한다. 부모의 실제 동결과 F parameter hash 검사는 부모 생성 실행기의 범위다.

`cache_fingerprint`는 실제 array·과학적 metadata를 결속하고, 읽기 경로에만 영향을 주는 `reused_prediction_cache` flag는 제외한다. 처음 생성한 cache와 같은 cache를 다시 읽은 경우 fingerprint가 같아야 한다. 반면 실제 부모 model SHA나 예측·mask·origin 등의 변경은 다른 fingerprint가 된다. `validate_completed_receipt`는 완료 상태·동일 run/variant·전체 source/config/data/scaler/model/parents 계약·private spec·필수 예측 family와 M/V split의 완결성·저장 예측 SHA·selected checkpoint SHA를 대조한다. 공통 `verify_contract`가 검사하지 않는 `parents_sha`도 명시적으로 검사한다. data SHA만 같은 다른 계약의 결과를 재사용하지 않는다.

비공개 `combinations/<run_id>/`에는 spec, CP0·학습 끝의 immutable full optimizer checkpoint, selected.pt, M/V 저장 예측을 남긴다. full checkpoint에는 모델·optimizer·sampler/permutation/RNG·누적 노출·최선 가중치·curve·계약이 포함된다. `spec.resume_checkpoint`에 full checkpoint를 넣고 **새 run_id**로 재개한다. 과거 디렉터리를 덮지 않는다. source/config/data/scaler/부모 fingerprint나 장치가 바뀌면 재개를 거부한다. run_id·resume 파일 위치만 scientific config hash에서 제외한다.

마감은 학습 batch 경계와 예측 microbatch 경계에서 확인한다. 예측은 창 캐시 생성 전에도 마감을 확인한다. 학습 중·V 평가 도중 마감이면 아직 평가하지 못한 endpoint와 그 segment의 batch loss를 full checkpoint에 추가해 저장하고 `BUDGET_LIMIT`로 반환한다. 만료 이후 새로운 M/V 추론을 시작하지 않고 부분 예측으로 유효 점수를 만들지 않는다. 이때 `scores=[]`, `prediction_artifacts={}`이며 CSV에는 빈 결과를 위한 header만 있다. CP0조차 평가하지 못한 경우 `selected_V_mse`와 `selected_state_sha`는 null이고 selected.pt를 만들지 않는다.

재개는 노출 수만으로 endpoint 완료를 판단하지 않는다. 완료된 V 평가 curve를 기준으로 미평가 endpoint를 다시 평가한다. 예를 들어 정확히 0.25 epoch의 마지막 update에서 마감이어도 0.25 endpoint V 평가를 건너뛰지 않는다. CP0 평가 도중 마감이면 재개 후 CP0부터 다시 평가한다. 부분 평가 결과는 버린다. 학습·선택이 끝난 뒤 최종 M/V 예측 저장 중 마감이면 마지막 완전 optimizer checkpoint에서 추가 update 없이 선택 상태와 예측 저장을 재개할 수 있다. 이 경로도 새 디렉터리에 immutable 재개 checkpoint를 남긴다. 이미 제출한 한 batch의 device 연산을 OS 수준에서 선점한다는 보장은 아니다.

공개 `<public_root>/<run_id>/`에는 평평한 curve.csv, scores.csv, V origin×channel×horizon-band SSE/SAE/count의 gzip CSV, receipt.json을 남긴다. 예측·실제 정답·원자료·checkpoint는 공개하지 않는다. 공개 영수증에는 상대 private cache reference와 hash만 들어간다. M 학습 전체 origin/고유 target 시점/관측 반복수·clipping·실제 최적화 시간·초기/최종/선택 가중치 SHA·변화량·head 파라미터·선정 CP를 남긴다.

이 담당 모듈의 F 호출수는0이다. 이것을 배포 F 비용0으로 해석하면 안 된다. conditional/alpha는 필요한 b와 g 전체 pipeline, past_control은 g+H, legacy_residual은 b+새R의 실제 호출을 root 비용 실행기가 측정해야 한다. 현재 process peak는 호출자가 이미 상주시킨 모델을 포함할 수 있으므로 fresh-process 배포 memory와 구분한다.

## 실제 확인한 검사

```text
python -m unittest test_combination test_combination_contract
```

위 명령은 해당 캠페인 디렉터리에서 실행한다. 최초 모듈 부재 실패(RED)를 확인한 뒤 구현했다. 결합 CPU 합성 14개 테스트는 다음을 확인한다: 관측수가 다른 채널의 FP64 alpha와 조밀 grid 대조·alpha 분모0, H/Hpast의 CP0=g와 동일 활성 용량·세 입력 block gradient, 채널 swap/미래까지 학습한 부모 거부, 실제 CPU toy 학습·저장 예측·checkpoint replay·부모 cache 불변, T_final 창 materialization 금지, Hpast의 저장 control flag 복원·replay 일치, 새 residual CP0=b와 current 중복 방지, 만료 예산 CP0 체크포인트 보존·재개, 실제 AdamW2 update 후 optimizer/sampler/RNG 재개와 무중단 모델 hash 일치다.

추가 failure injection은 transient flag만 바뀐 동일 cache의 identity 유지, 실제 부모 model 변경과 실제 저장 NPZ 손상 거부, 만료된 CP0의 추론 호출0, CP0 예측 두 microbatch 후 마감과 부분 점수 폐기, 학습 완료 뒤 최종 예측 저장 마감의 재개를 확인한다. 완료 상태를 잘못 재사용한 경우 및 CP0/마감 예측 경로의 기존 실패를 먼저 실제로 재현하고 수정했다. 재개한 모델의 last/selected SHA는 무중단 학습과 일치했다.

```text
결합 범위: Ran 14 tests in 2.331s / OK
새 캠페인 전체 CPU 회귀: 114 passed, 26 subtests passed in 6.24s
```

전체 CPU 회귀는 GPU를 노출하지 않고 BLAS/PyTorch 2 thread로 실행했다. 해당 캠페인 `test_*.py` 전체 범위이며 저장소의 다른 역사적 실험 전체를 다시 실행한 것은 아니다.

합성 검사 PASS를 실제 자료의 학습 완료·수렴·개선·원격 게시로 바꾸어 말하지 않는다. 실제 성능·별도 FP64 재채점·부모 F replay·T 최종 평가는 캠페인 결과의 별도 상태다.
