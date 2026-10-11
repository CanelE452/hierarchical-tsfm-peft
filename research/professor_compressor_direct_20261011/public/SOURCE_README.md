> 게시용 사본입니다. 실험 완료 당시 원본 문서의 SHA256은 `dd88f21ed8987aec8d4b68ddc2894ee021a5c3c7b13e07ed864f3ed9c926c685`입니다. 본문 수치·판단을 유지하고 링크만 GitHub 경로로 조정했습니다. 원문의 “이번 작업에서 push하지 않음”은 과학 실험 완료 당시 기록이며, 후속 사용자 승인으로 이 공개 사본을 게시합니다. 구 TEST 사고는 [범위·원본 SHA](PROTECTED_ACCESS_CONTEXT.json)만 공개하며 보호된 통계는 재게시하지 않습니다.

# 교수님 Compressor 직접 검증의 소스·재현 안내

이 폴더의 목적은 GTConvAE와 TS2Vec를 실제로 학습하고 TSFM 뒤 대응하는 decoder로 원공간 미래를 복원하는 것이다. 이전 LEVEL 중심 실험의 K5/K6, D8, 잔차 G, 잠재 자기예측 손실은 이번 구조의 필수조건이 아니다. 코드가 존재하거나 CPU fixture가 PASS라는 사실과 실자료 학습·미래 추론 완료를 구분한다. 실행된 조건은 [runs.json](../runs.json), 실제 수치는 [metrics.csv](../metrics.csv), 연구적 해석은 [REPORT_KO.md](REPORT_KO.md)에서 확인한다. **전체 실행의 최종 상태는 별도 `FINAL_EXECUTION_AUDIT.json`의 실제 coverage·검산·비용·자원 항목으로 확인한다.** 그 파일이 없거나 필수 항목이 미완료면 전체 완료로 해석하지 않는다. 이 안내문에는 갱신 중인 완료 수를 고정하지 않는다.

## 원문·공식 소스와 적용 코드

- GTConvAE: [논문](https://proceedings.mlr.press/v198/sabbaqi22a.html), [PDF](https://proceedings.mlr.press/v198/sabbaqi22a/sabbaqi22a.pdf). [models_gt.py](../models_gt.py)는 원문 기반 재구현이다. product graph·filter bank·시간 압축·native GT decoder를 유지한다. Figure1(B)와 본문의 decoder 연산 순서 차이 및 signed 자료의 최종 linear 출력은 [GT_METHOD_DELTA.md](GT_METHOD_DELTA.md)에 기록했다. 확보하지 못한 공식 코드를 확보한 것처럼 표시하지 않는다.
- TS2Vec: [공식 revision b0088e14](https://github.com/zhihanyue/ts2vec/tree/b0088e14a99706c05451316dc6db8d3da9351163). 기존 vendor의 7개 알고리즘 파일은 SHA를 확인한 뒤 수정 없이 재사용한다. [ts2_adapter.py](../ts2_adapter.py)는 import/export와 이번 별도 reconstruction decoder를 연결한다. hidden64/depth10·d320 API 기본값과 d16 적용, offline E→F→D와 causal sliding last→native ridge의 차이는 [TS2_METHOD_DELTA.md](TS2_METHOD_DELTA.md)에 있다. 공식 forecasting/CLI/ridge 참고 소스는 [vendor_reference/ts2vec/provenance.json](../vendor_reference/ts2vec/provenance.json)에 근거를 기록한다.
- TSFM: [Chronos 공식 저장소](https://github.com/amazon-science/chronos-forecasting). [foundation.py](../foundation.py)는 설치된 **Chronos-Bolt-small**의 직접 median quantile 경로를 사용하며 공식 pipeline과 실제 parity를 확인한다. 모델 cache revision, installed source SHA, 최대 context와 native horizon은 실행 시 `FOUNDATION_PROVENANCE.json`에 기록한다. 다른 Chronos 계열의 API를 섞지 않는다.
- 데이터: [data_io.py](../data_io.py)는 기존 v18의 TRAIN/VAL archive와 관측 mask만 읽는다. `include_test=False`를 사용하고 TEST archive는 열거나 hash하지 않는다. [DATA_PROVENANCE.json](../DATA_PROVENANCE.json), [DATA_CHECKS.json](../DATA_CHECKS.json), [DATA_NOTE_KO.md](DATA_NOTE_KO.md)에 원자료·분할·정규화·graph·배열 hash·실제 창 수를 기록한다.
- 운영·기록: [common_direct.py](../common_direct.py)의 CPU/GPU/RSS/저장량 누적 상한과 [RESOURCE_BUDGET.json](../RESOURCE_BUDGET.json)을 사용한다. 이전 실행의 자원 사용분을 포함한다. `run_support.py`는 실제 prediction/diagnostic NPZ를 별도 cache에 저장하고 source pins를 result에 기록한다. weight/raw archive를 공개하거나 Git commit/push하는 코드는 이 작업 범위에 포함하지 않는다.

## 학습과 직접 비교

아래 CLI는 사용한 구조와 실행 인자를 재현하기 위한 기록이다. **완료된 canonical ID를 그대로 다시 학습하지 않는다.** 실행 전 [protocol.json](../protocol.json), [runs.json](../runs.json), [EXTRA_ACTUAL_RUNS.json](../EXTRA_ACTUAL_RUNS.json), [EXTRA_FAILED_ATTEMPTS.json](../EXTRA_FAILED_ATTEMPTS.json)과 실제 GPU 점유·누적 예산을 확인한다. 신규 재실험은 새 run ID/attempt와 별도 출력 경로가 필요하다. GT runner는 같은 cache 경로를 재사용할 수 있으므로 기존 경로가 있는 아래 명령을 그대로 재실행하면 안 된다. 독립 GPU 학습·추론·검산은 합쳐 **동시1개**다.

GT native 구조의 명령 형식:

```powershell
.venv\Scripts\python.exe research\professor_compressor_direct_20261011\runner_gt.py --dataset robin --seed 92601
.venv\Scripts\python.exe research\professor_compressor_direct_20261011\runner_gt.py --dataset robin --seed 92601 --no-spatial
```

정상 크기 TS2Vec의 Robin d16/seed92601은 epoch0 decoder selection을 포함한 새 ID로 다시 학습했다. 실제 사용한 ID의 CLI는 다음과 같으며, 원래 pilot은 별도로 보존한다.

```powershell
.venv\Scripts\python.exe research\professor_compressor_direct_20261011\runner_ts2.py --dataset robin --seed 92601 --dim 16 --retry-tag epoch0_v2 --encoder-batch 8
```

Jena의 현재 protocol은 `memory_retry.py`와 `batch4_v4` 새 ID를 사용한다. d320/seed92601에 해당하는 실제 CLI 형식:

```powershell
.venv\Scripts\python.exe research\professor_compressor_direct_20261011\memory_retry.py --dataset jena --seed 92601 --dim 320 --retry-tag batch4_v4 --encoder-batch 4
```

[memory_retry.py](../memory_retry.py)는 **PyTorch import 전** 해당 child process에 `PYTORCH_ALLOC_CONF=backend:native,garbage_collection_threshold:0.6`, `TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=128`을 설정하고 이전 alias `PYTORCH_CUDA_ALLOC_CONF`를 제거한다. 설정·실제 명령·wrapper/runner SHA는 `execution_environment/<id>.json`에 남긴다. 모델 크기와 공식 loss는 유지하지만 batch4는 negative batch 구성과 sample exposure를 바꾸므로 정확한 공식 기본 batch8 재현으로 표시하지 않는다. 이전 batch8 및 cache-only retry의 실패와 부분 weight는 보존한다. 실패의 직접 guard는 기록된 RSS/allocated 값으로 판단하며 CUDA reserved 크기만으로 allocated VRAM 초과를 단정하지 않는다.

[확인] 공식 `fit()`의 실제 DataLoader batch는 `min(configured_batch, instance_count)`다. Robin TRAIN `[1,1992,17]`은 max_length3000 split을 하지 않아 **실제 B1**이고, 공식 `instance_contrastive_loss`는 B1에서 **0**을 반환한다. temporal contrastive 항은 별도로 남는다. Jena TRAIN `[1,26207,21]`은 공식 규칙으로 8개 section으로 나뉘고 현재 적용은 **B4**다. 따라서 두 자료의 실제 instance negatives 조건이 다르다. 200/600 update, hidden64/depth10, d16/d320과 같은 모델 표기만으로 이 차이를 숨기지 않는다. 근거는 pinned vendor의 [fit 경로](../../level_k_tradeoff_20261011/vendor/ts2vec/ts2vec.py)와 [loss 경로](../../level_k_tradeoff_20261011/vendor/ts2vec/models/losses.py), 각 실제 run의 `initial_protocol.json`/encoder configuration이다.

이 명령을 안내했다는 사실은 새 실행을 승인하거나 실행했다는 뜻이 아니다. 기존 결과를 덮지 않는 새 run ID/attempt 규칙은 실제 runner/프로토콜을 따른다. TS2Vec의 SWA save/load는 inference state 복원이므로 optimizer·iteration 자동 재개로 해석하지 않는다. 프로토콜·checkpoint·데이터가 달랐던 예전 graph-inspired/D8 결과는 이 native 실험의 재사용 근거가 아니다.

GT는 reconstruction E/D를 학습한 checkpoint A와, E 및 공유 product 계수를 고정하고 같은 native D 필터만 실제 F TRAIN 예측에 적응한 checkpoint B를 구분한다. TS2Vec는 원래 encoder 이후 새로 학습한 nonlinear D를 사용하며 같은 A/B 구분을 따른다. 실제 미래 label을 사용한 calibration B의 이득을 비지도 표현학습만의 이득으로 표시하지 않는다.

비교는 원채널 Raw F, mean-block 시간 축소→F→repetition, 각 E→F→D transfer/calibration, latent 마지막값 유지→같은 D, TS2 native ridge, GT 공간 제거 재학습이다. d320은 표현 확장이고 GT는 시간 scalar 감소와 F 행 수 증가가 함께 존재한다. reconstruction MSE/MAE와 future MSE/MAE는 별도 metric이다. 두 seed/반복 개발 VAL만으로 유의성·일반화·신규성을 확정하지 않는다.

## 변경 전 소스와 protocol 버전

기존 결과의 source SHA가 현재 파일과 달라지면 원래 소스 snapshot의 hash와 연결해 검산한다. 수정 전 결과를 현재 implementation으로 실행한 것처럼 바꾸지 않는다. [source_snapshots](../source_snapshots)에는 GT guard 수정 전 runner, TS2 epoch0 수정 전 runner, dispatcher 수정 전 소스가 보존되어 있다.

```text
버전   역할                                보존/현재 경로
v1     최초 실행 계획                      source_snapshots/protocol_epoch0_v2/protocol_v1.json
v2     TS2 epoch0 selection 포함           source_snapshots/protocol_memory_retry_v3/protocol_v2.json
v3     Jena cache-only 운영 재시도          source_snapshots/protocol_batch4_v4/protocol_v3.json
v4     Jena batch4 적용 및 실패 ID 보존      protocol.json
```

각 버전의 실제 파일과 hash를 확인하며 v4를 과거 run의 원래 protocol로 소급 표시하지 않는다. 현재 canonical ID는 v4 `protocol.json`을 따른다. [이전 목표 전환 기록](../OLD_FIT_CANCELLED.json)과 [목표·방법 변경](GOAL_AND_METHOD_DELTA.md)에 따라 기존 LEVEL/A–H queue, PCA/D8/G 경로를 자동 재개하지 않는다.

## 저장된 실제 결과에서 보고서와 그림 다시 만들기

```powershell
.venv\Scripts\python.exe research\professor_compressor_direct_20261011\analyze_direct.py
```

[analyze_direct.py](../analyze_direct.py)는 **모델/weight를 load하지 않고 GPU 작업 없이** 새 direct-study의 `runs/*/result.json`, `baselines/*/result.json`과 이들이 지정한 saved prediction/diagnostic NPZ를 읽는다. 원자료 loader를 다시 호출하지 않는다. result와 NPZ SHA, 데이터 계약, 동일 VAL origin/target/mask, 독립 NumPy observed MSE/MAE를 확인한다. 잘못된 경로는 별도로 실패 기록을 남기고 다른 유효한 결과의 보고를 계속한다. 실제 결과가 0개일 때는 `NO_ACTUAL_RESULTS_YET`로 표시하고 수치를 만들어내지 않는다.

Canonical 보고서·metric·16fit coverage·seed 평균에는 `protocol.json`의 명시 `jobs[].id`와 `result.job.id`가 일치하고 dataset/seed/family/dim 또는 spatial 조건도 같은 run만 들어간다. 예를 들어 epoch0 decoder selection을 수정한 `robin_TS2_d16_s92601_retry_epoch0_v2`가 protocol의 요구 ID라면 원래 `robin_TS2_d16_s92601` 완료 결과로 그 자리를 채우지 않는다. 새 ID가 완료될 때까지 그 조건은 미완료다. 원래 ID는 `canonical_id`로 연결하며 수정 전 실제 결과·hash·비용을 별도 manifest에 보존한다. 수정 전 run과 retry를 두 seed로 세거나 평균에 함께 넣지 않는다. protocol ID가 없거나 중복이면 임의로 run을 고르지 않는다. 실제 실행 중 여부도 retry tag를 포함한 정확한 ID로 연결한다.

완료 result가 없는 조건에 `runs/<id>/failure.json`이 있으면 `FAILED_PRESERVED_PATH`로 표시한다. 새 cache의 `initial_protocol.json` 또는 learning curve만 있고 실제 실행 프로세스가 없으면 `PARTIAL_ARTIFACTS`로 표시한다. 현재 ID의 live 프로세스가 있으면 `RUNNING_CONFIRMED_PROCESS`다. 이 상태들은 `not_started`나 완료 fit으로 묶지 않는다. 실패 원래 ID가 protocol의 새 batch/retry ID로 교체되어도 실패 receipt·partial iteration·예정 iteration·batch·RAM/allocated VRAM·재개 안내를 계속 보존한다. 부분 SWA checkpoint를 optimizer resume나 완료 encoder로 가정하지 않는다.

생성 파일:

- `metrics.csv`: reconstruction과 forecast 분리, seed별 값, Raw 대비 MSE/MAE 변화율, 채널별·horizon별 오차, result에 기록된 물리 단위 채널 MAE.
- `runs.json/csv`: replacement의16개 representation fits와 실행·미실행·invalid 상태. baseline 두 자료는 별도로 센다.
- `EXTRA_ACTUAL_RUNS.json`: 현재 protocol ID 집계 밖의 실제 run ID, 원래 result SHA·수치·비용·소스·제외 이유와 추가 완료 actual run receipt 수. epoch0 selection 수정 전 pilot 수치는 보존된 원래 receipt 값이며 canonical 독립 예측 검산/seed 평균에 포함하지 않는다. 기존 actual result·NPZ·그림 파일은 삭제하거나 덮지 않는다.
- `EXTRA_FAILED_ATTEMPTS.json`: 현재 protocol과 retry 이전 ID의 실패 attempt 및 실행 프로세스 없이 남은 부분 artifact. failure/protocol/curve SHA, 실제 또는 저장 curve 하한 iteration, 예정 iteration, batch, resource receipt와 이유를 기록한다. 이 manifest를 만들 때 weight·모델·원자료를 읽지 않는다. 성공 결과만 보존하고 실패 실행을 빠뜨리지 않는다.
- `REPORT_KO.md`: Q1 표현/복원, Q2 E→F→D, Q3 동일 D의 calibration, Q4 GT 공간 연결을 실제 확보된 결과 범위에서 설명한다. 첫 실제 fit 이후 `INTERIM_RESULTS_KO.md`도 즉시 갱신한다.
- `figures/*_forecast.png`: error를 보기 전에 정한 **origin row0/middle/last와 channel0/middle/last**의 target/Raw/mean-block/native transfer/calibrated/persistence/native ridge 예측. 관측되지 않은 target은 NaN gap이며 보간값을 관측 정답으로 그리지 않는다.
- `figures/*_reconstruction.png`: 같은 고정 origin의 과거 원본과 reconstruction-trained D 출력. 그림의3예시 오차를 full-VAL reconstruction metric과 구분한다.
- `figures/*_latent_curves.png` / `*_latent_heatmap.png`: 첫8개 고정 좌표와 전체 좌표·시간을 표시한다. d320 expansion을 표시하며 좌표 크기/군집 미관을 정확도 증거로 해석하지 않는다.
- `figures/*_horizon_errors.png`: 모든 실제 VAL origins의 observed channel-macro horizon MSE/MAE.
- `FIGURE_MANIFEST.json` / `ANALYSIS_VERIFICATION.json`: 그림이 사용한 실제 입력/source SHA·truth contract·고정 선택, 저장 예측 metric 검산 및 CPU/RSS 자원. 동일 hash의 이미 만든 그림은 다시 그리지 않는다.

## 전체 재구성·strict 검산·한눈 비교 그림

저장 forecast NPZ의 모든 VAL origins 검산과, 원래 A checkpoint로 과거512 전체를 새로 재구성하는 검산은 별도다. [verify_full_reconstruction.py](../verify_full_reconstruction.py)는 GT joint native E/D A 및 TS2 SWA E + reconstruction-best D A만 사용한다. GT forward B16, TS2 offline export B8/D B32, FP32·TF32 false를 유지한다. CPU 계약 확인 → GPU 새 전체 VAL 재구성 → CPU 독립 metric/새 archive의3단계를 계측한다. 새 학습·F 호출은0이며 TEST archive는 열지 않는다. 현재 canonical GT8/TS2 8개를 요구하며 원래 reconstruction macro/channel/horizon/physical metric과 독립 NumPy 계산을 비교한다.

다음 명령은 GPU 추론을 수행하므로 다른 GPU job이 끝난 뒤 누적 예산 내에서 한 번에 하나씩 실행한다.

```powershell
.venv\Scripts\python.exe research\professor_compressor_direct_20261011\verify_full_reconstruction.py --family GT
.venv\Scripts\python.exe research\professor_compressor_direct_20261011\verify_full_reconstruction.py --family TS2 --dataset robin
```

Jena 재구성 검산에도 새 child process의 PyTorch import 전 운영 설정을 적용할 수 있다. 학습용 `memory_retry.py`는 runner를 재학습하므로 이 검산에 대신 호출하지 않는다. 저장소 root에서 실행하는 검산 전용 명령:

```powershell
.venv\Scripts\python.exe -c "import os, runpy, sys; os.environ.update({'PYTORCH_ALLOC_CONF':'backend:native,garbage_collection_threshold:0.6','TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT':'128'}); os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None); sys.path.insert(0, 'research/professor_compressor_direct_20261011'); sys.argv = ['verify_full_reconstruction.py', '--family', 'TS2', '--dataset', 'jena']; runpy.run_path('research/professor_compressor_direct_20261011/verify_full_reconstruction.py', run_name='__main__')"
```

audit 출력은 `.cache/professor_compressor_direct_20261011/reconstruction_audit/<id>/full_VAL_reconstruction.npz`, `reconstruction_audit/<id>/receipt.json`과 새 timestamp summary다. 기존 PASS audit는 원래 result/checkpoint/NPZ hash·독립 metric을 다시 확인해 재사용한다. 실패·부분 audit는 보존하고 새 `--audit-tag TOKEN`으로 별도 경로에 재시도한다. `--job-id EXACT_ID`를 반복해 선택할 수 있지만 전체 완료는 canonical16개의 receipt가 있어야 한다.

전체 저장 예측·cost·diagnostic·source snapshot·checkpoint freeze·TS2 protocol을 CPU에서 strict 검산하는 명령:

```powershell
.venv\Scripts\python.exe research\professor_compressor_direct_20261011\verify_direct.py --strict
```

[verify_direct.py](../verify_direct.py)는 TRAIN/VAL 자료·저장 NPZ·필요한 CPU checkpoint를 읽고 새 `checkreceipts/direct_verification_<timestamp>.json`을 남긴다. 과학 결과와 기존 receipt를 수정하지 않고 모델을 새로 학습하거나 F/GPU 추론을 호출하지 않는다. strict는 모든 planned run·baseline의 실제 완료와 artifact 검산을 요구하므로 부분 실행에서는 nonzero 종료가 정상이다. full reconstruction 새 추론 receipt와 이 strict receipt를 함께 최종 실행 audit에서 확인한다.

16개의 canonical 저장 예측 검산이 모두 PASS한 뒤 2자료×MSE/MAE 한눈 비교 그림을 만든다.

```powershell
.venv\Scripts\python.exe research\professor_compressor_direct_20261011\generate_summary_fig.py
```

[generate_summary_fig.py](../generate_summary_fig.py)는 저장 metric CSV와 protocol/검산 metadata를 사용해 Raw/mean-block/GT full/no-spatial/TS2 d16/d320을 고정 순서로 표시한다. A/B/native 경로는 색·marker로, 두 seed 원수치는 filled/hollow 및 각 패널 표로 표시한다. Raw 기준선을 넣고 CI/유의성을 표시하지 않는다. [한눈 비교 그림](../figures/METHOD_COMPARISON.png)과 [그림 manifest](../METHOD_COMPARISON_MANIFEST.json)에 CSV/source/verification SHA·고정 규칙·CPU 자원을 기록한다. 비용·메모리는 각 fixed resident prototype process 측정 범위이며 isolated deployment footprint 절감 증거로 확대하지 않는다.

## 로컬 자료와 운영 범위

현재 원자료·weights·prediction NPZ는 로컬 cache에 있으며 GitHub에 게시되었다고 가정하지 않는다. 실행 전에 metadata가 가리키는 실제 경로와 hash를 확인한다.

```text
자료/산출물             저장소 root 기준 로컬 경로
Robin TRAIN/VAL         .cache/tsfm_peft_level_confirmation_v6_20260928/data/trainval.npz
Jena TRAIN/VAL          .cache/tsfm_peft_practical_controls_v7_20260928/data/jena_trainval.npz
Chronos cache           .cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8
학습 weights/원 예측    .cache/professor_compressor_direct_20261011/runs/<canonical-id>/
GT 초기 수정 attempt    위 run 경로의 attempts/contiguous_guard_v2/
전체 재구성 검산 배열   .cache/professor_compressor_direct_20261011/reconstruction_audit/<canonical-id>/
```

weights의 정확한 A/B/encoder 경로는 각 `runs/<id>/result.json`의 training/calibration에 있다. source·CSV·보고서·그림의 GitHub 상대 링크와 로컬 cache 경로를 구분한다. 원자료의 provenance는 [DATA_PROVENANCE.json](../DATA_PROVENANCE.json), 모델 revision은 [FOUNDATION_PROVENANCE.json](../FOUNDATION_PROVENANCE.json)에 보존한다.

[RESOURCE_BUDGET.json](../RESOURCE_BUDGET.json)의 누적 GPU8h·CPU12h·wall12h·RAM8GiB·allocated VRAM8GiB·저장16GiB 상한에는 이전 사용량을 포함한다. GPU 동시 실행1개, 보호된 TEST 새 접근0, 승인 없는 commit/push·논문/PPT 작성0 범위를 유지한다. 이 문서에 재현 CLI를 추가하는 작업은 학습·GPU·TEST·자동 queue 재개를 실행하지 않는다.

빈 실행·compile·CPU tiny fixture는 실자료 fit이나 미래 성능에 포함하지 않는다. 전체16fit·두 자료 baseline·필수 future 경로·비용·전체 재구성·strict 검산의 완료 여부는 별도 `FINAL_EXECUTION_AUDIT.json`에서 확인하며 [REPORT_KO.md](REPORT_KO.md)는 그 실행 범위에서 연구적 해석을 제시한다. 최종 audit 파일이 아직 없으면 예정 파일명을 완료 증거 또는 유효한 링크로 취급하지 않는다. 문서/그림은 상대 경로를 사용하며 이번 요청에서는 push하지 않는다.
