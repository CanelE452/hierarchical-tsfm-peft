> 게시용 사본입니다. 실험 완료 당시 원본 문서의 SHA256은 `44987377531427b9be1b1ef2d35e213cb416985e92ce6e913d37f8949fcef2f1`입니다. 본문 수치·판단을 유지하고 링크만 GitHub 경로로 조정했습니다. 원문의 “이번 작업에서 push하지 않음”은 과학 실험 완료 당시 기록이며, 후속 사용자 승인으로 이 공개 사본을 게시합니다. 구 TEST 사고는 [범위·원본 SHA](PROTECTED_ACCESS_CONTEXT.json)만 공개하며 보호된 통계는 재게시하지 않습니다.

# 교수님 압축기 직접 실험: 독립 소스 감사

2026-10-11. 감사 범위는 새 `professor_compressor_direct_20261011` 목표의 실행 소스와 당시 존재한 JSON 결과·영수증이다. 기존 A–H 큐는 이 감사에서 재개하지 않았다. 과학 소스·모델·결과를 수정하지 않았고, 모델 학습·GPU 추론·CUDA 초기화·TEST 파일 열람·원본 데이터 열람을 수행하지 않았다. 이 문서만 새로 작성했다.

## 판단

[확인] 현재 실행 소스에는 실제 미래 정답을 E/F 추론 입력으로 전달하는 경로가 보이지 않는다. GT는 native graph-time E/D를 유지하며, TS2Vec는 정상 크기의 공식 encoder와 별도 비선형 복원기를 사용한다. 두 경로 모두 실제 frozen Chronos-Bolt 출력을 받아 원 채널 미래를 복원한다. decoder calibration은 TRAIN 미래 label로 학습하는 별도 감독 단계이며, 비지도 압축기만의 효과와 구분해야 한다.

[확인] 구현 경로가 있다는 것과 모든 16개 조건을 실제로 완료했다는 것은 다르다. 감사 당시 GT full 두 개와 TS2 결과 세 개의 JSON을 읽었으며, TS2 세 개 중 하나는 epoch0 후보 누락을 보존한 pilot이다. 이것은 최종 진행률 산정이 아니다. 당시 읽은 `checkreceipts/direct_verification_1791723008690944400.json`도 saved-artifacts PASS와 campaign_complete=false를 별도로 기록했다. 이 감사는 예측 NPZ를 직접 재계산하지 않았다.

## 실행 경로 확인

- **GT native E/D**: `models_gt.py`의 polynomial filter, 공유 product coefficients, 시간 down/up-sampling을 양쪽에 유지한다. L512 입력은 `[B,128,C,2]`가 된다. `runner_gt.py:226`–`227`은 TRAIN/VAL 과거 latent에 실제 F를 호출하고, `decode_future`는 과거 latent와 예측 latent 12개를 연결한 같은 native decoder에서 원시 미래 48개만 crop한다. raw C 채널 자체는 줄이지 않으며 F 입력 행은 2C개다. scalar 절반·시간 압축을 raw-channel 개수 축소와 혼동하면 안 된다.
- **GT calibration**: `runner_gt.py:131`–`138`은 전체 모델을 먼저 freeze한 뒤 native D filters만 trainable로 만든다. optimizer에는 `decoder_parameters()`만 들어가며 E와 공유 product의 state를 실제 비교한다(`:185`–`187`). 원래 reconstruction model을 deep copy하므로 A/B가 같은 native D의 출발점을 공유한다. reconstruction-transfer epoch0도 후보로 보존한다. F 출력 `ftrain`은 이미 실제 F에서 생성한 TRAIN 예측이고, 정답 미래 latent를 대신 쓰지 않는다.
- **TS2 정상 설정**: `ts2_adapter.py:71`–`85`는 공식 클래스를 hidden64/depth10, d16/d320, lr0.001, batch8, max_train_length3000으로 구성한다. `runner_ts2.py:244`–`272`는 연속 TRAIN 배열에서 공식 default `fit()`을 실제 호출하고 200/600 update 완료를 검사한다. d320은 Robin C17/Jena C21에서 표현 확장이다. 이를 경량 압축이라고 부를 수 없다.
- **TS2 E/F/D**: `runner_ts2.py:287` 이후 online/SWA encoder를 freeze한다. 각 origin의 과거 512점에서 offline representation을 export하고, `:319`–`338`에서 실제 F 미래 48개를 생성한다. `:340`–`347`은 원 decoder를 복사해 동일 구조·초기 state에서 decoder만 calibration한다. 원 reconstruction decoder state를 실제 비교하고, canonical retry에는 epoch0 후보가 포함된다.
- **TS2 native ridge**: native 대조는 main offline export와 다르게 공식 causal sliding 설정의 마지막 201점 representation을 사용한다. `runner_ts2.py:349` 이후 full TRAIN stride1 origins의 representation/observed future labels로 ridge를 fit하고 기존 개발 VAL에서 alpha를 고른다. 별도 sklearn 수학 fixture, 실제 trained encoder의 official rolling export와 수치 비교를 실행하도록 되어 있다. calendar covariates·원 benchmark split을 그대로 재현한 대조라고 주장하지 않는다.
- **실제 F**: `foundation.py:16`–`24`가 설치된 pinned Chronos-Bolt-small을 FP32로 로드하고 eval/freeze한다. `:41`–`53`은 latent의 time 축을 보존하고 각 좌표를 독립 batch row로 만들어 실제 model forward/median 미래를 반환한다. synthetic ramp의 공식 pipeline parity는 API 경로 진단이며 실자료 예측 품질의 근거가 아니다.
- **동일 정답과 마스크**: `data_io.py`는 parent `load_data(..., include_test=False)`를 호출한다. parent `runtime_v18.py:108`–`114`에서 TRAIN/VAL archive와 TEST archive를 선택하는 분기를 확인했다. 평균·표준편차와 spatial graph는 TRAIN 관측만 사용하고 input은 causal forward-fill한다. reconstruction과 미래 예측은 서로 다른 windows/labels로 평가된다. `common_direct.py:78`–`104`는 original observed mask를 유지하고 각 원채널 MSE/MAE를 계산해 macro 평균한다. 학습 loss는 observed-entry mean이므로 평가 macro와 목적 함수 가중치는 다르며 문서에 명시해야 한다.
- **B1/B32 비용**: `run_support.py:14`–`58`은 physical FP32 past에서 normalization→H2D→전체 method→D2H/sync를 실제 호출한다. saved full-VAL prediction과 parity 후 3 warmup/10 timed를 수행한다. GT는 transfer/calibrated 각 두 batch, TS2는 native ridge까지 각 두 batch를 현재 결과에 기록했다. Robin B32는 30 distinct origins와 두 반복을 기록하며 10 timing 반복을 정확도 표본으로 세지 않는다.

## 구체적인 검산 개선점

### 1. GT epoch0를 정상 선택해도 독립 verifier가 거짓 FAIL할 수 있음

[확인] `verify_direct.py:820`은 calibration D state가 반드시 바뀌어야 PASS라고 요구한다. 하지만 `runner_gt.py:140`–`144`는 원 reconstruction decoder를 epoch0 후보로 저장한다. epoch0가 최선이면 D가 그대로인 것은 올바른 결과다. 현재 읽은 GT 두 개는 epoch50이 선택되어 이 조건에 걸리지 않았다.

최소 조치: selected_epoch=0이면 D unchanged와 transfer/calibrated prediction 동일성을 검사하고, selected_epoch>0이면 D 변경 여부를 별도로 진단한다. optimizer update가 존재한다는 사실과 선택된 checkpoint가 변경됐다는 사실을 구분한다. 독립적인 다른 조건 실행을 중단할 이유는 없다.

### 2. 비용 coverage가 존재하는 method 목록으로부터 만들어짐

[확인] `verify_direct.py:388`–`398`의 required pairs는 실제 저장된 cost_methods에서 생성된다. 따라서 TRANSFER_F 두 batch만 있어도 CALIBRATED_F 비용 누락을 coverage PASS로 볼 수 있다. 현재 읽은 GT/TS2 완료 결과에는 실제 필수 4/6 rows가 존재했다.

최소 조치: family별 고정 필수 목록을 사용한다. GT는 TRANSFER_F/CALIBRATED_F × B1/B32, TS2는 여기에 TS2_NATIVE_RIDGE × B1/B32다. latent-persistence 비용은 현재 프로토콜의 필수 대상이 아니므로 forecast 전체를 무조건 비용 필수로 만들지 않는다.

### 3. F frozen과 누출 guard의 자기보고 한계

[확인] `foundation.py:66`–`69`의 unchanged는 실제 state hash 비교이며 parameter_gradients_absent도 실제 parameter 상태를 검사한다. 반면 F_backward_calls는 독립 hook을 연결한 측정값이 아니라 Resources의 초기 counter를 반환한다. 현재 no_grad/freeze 실행 경로가 더 강한 근거다. counter 0만으로 독립 backward 검산을 했다고 표현하면 안 된다.

[확인] GT `runner_gt.py:261`은 F unchanged 결과를 guard에 저장하지만 false일 때 즉시 실패시키지 않으며 비용 측정 이전 검사다. TS2 `:419`–`423`은 비용 이후 encoder/F state 또는 gradient 문제가 있으면 실패시킨다. 현재 읽은 GT guard 두 개는 unchanged/grad-absence=true였다. GT도 마지막 비용 이후 실제 hash/grad 결과를 assert하면 실행과 완료 상태가 일치한다.

[확인] 두 runner의 미래 perturbation은 미래 부분을 바꾼 뒤 동일한 과거 slice를 두 번 실제 inference에 전달하는 검사다. 과거만 받는 API의 반복 수치 일치는 확인하지만, data loader·scaler·graph 생성까지 포함한 독립 누출 증명은 아니다. 소스 추적상 scaler/graph가 TRAIN만 사용하고 origin 이후 정답을 inference input으로 넘기지 않는 사실과 함께 설명해야 한다.

## 비교 해석의 범위

[확인] A/B 및 F/latent-persistence 비교는 각 case 안에서 같은 decoder를 사용한다. GT와 TS2 사이의 decoder architecture·latent 길이·좌표 수까지 모두 동일한 통제 실험은 아니다. calibration 이득에는 TRAIN 미래 감독이 추가된다. native ridge도 유효 history201, 더 많은 stride1 TRAIN origins, VAL alpha 선택을 사용하므로 main E/F/D의 decoder만 대체한 동일 조건 비교라고 부르지 않는다.

[확인] 비용 메모리는 격리된 단일 배포 모델의 최소 footprint가 아니라 **그 시점에 resident한 prototype 전체**다(`run_support.py:54`). TS2는 CPU online encoder·CUDA SWA encoder·두 decoder·F를 보유하며 native ridge timing 때도 이미 로드한 F와 두 decoder가 남아 있다. GT도 A/B 두 모델이 resident하다. full-path latency를 실제 측정한 사실은 유지되지만, 이 peak만으로 method별 최적 배포 메모리나 경량화 달성을 주장할 수 없다.

[확인] 모두 반복 노출된 개발 VAL이다. GT reconstruction checkpoint, D calibration checkpoint 및 ridge alpha가 VAL로 선택된다. 모델 구성별 결과가 좋거나 나쁜지 판단하는 탐색 자료로 사용할 수 있으나 독립 TEST 일반화나 두 seed만으로 강한 우월성을 확인했다고 표현하지 않는다. 기존 Electricity 보호 접근 incident는 새 경로의 TEST 접근 0 주장과 별개로 남아 있다.

## 읽은 소스 snapshot SHA256

```text
runner_gt.py       daddb757574904837bd80a3cd8042a9fc707989a83cbb3d9c252290b572c3339
runner_ts2.py      42e0241b437fa388ff6efd0015c553ae6a929c516ca0310f006c43421c817597
foundation.py      7fb2b5f6ddeb54ce3ec2057e4aa50f65e9efff1e5386d0b1eaa0a64035e8c05a
run_support.py     f32acf5bca736e63c00462798d04171fcfbd5e0a74599682cc976dbd9e9bc948
models_gt.py       a4f181a3c030cdc6acc17b784f3773648fe277bd0220790c1a7935814cf4039a
ts2_adapter.py     9a8d0571b5a6f5908d6db6fd8399c8d9afb815d716b3194faa4df4d4c27cafb4
data_io.py         c1a1ff96521e8a7b1e7121f94517e039209d47f461e2a6ca68ba5c69de3d432e
common_direct.py   b4a5305ab3d3af7082eb7e13ef8735516ae2171b3f69c8245c103c04f2dfe8dd
protocol.json      d8da7c161c75a55d0e5c15b1c9a1861d352472720c4e5fdb57a571dc82a08967
verify_direct.py   e94f240439665637afb04ddca96a808f5e3ea09e776ae09bae549337a07cc922
```

이 SHA는 감사 중 읽은 파일의 식별값이며 실행 중인 Python 프로세스가 로드한 모든 모듈을 독립 증명하는 값은 아니다. 이후 소스/결과가 추가되면 최종 상태를 별도 검산해야 한다.
