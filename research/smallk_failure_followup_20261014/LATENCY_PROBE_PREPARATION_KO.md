# D3 동일 조건 지연 분해 준비

상태: **SOURCE_READY / 실제 과학 데이터·모델·GPU 미실행**. 이 문서 작성 단계에서는 기존 Python 소스와 준비 JSON만 읽고 synthetic CPU 배열로 검사했다. 실제 속도 개선, 실제 출력 parity, 장비 성능은 아직 확인하지 않았다.

## 고정 비교

원 `smallk_predictive_alignment_20261013/controls.py`의 `deployment`가 기준이다. 이 경로는 `experiment.deployment`와 달리 MAIN의 현재 상태 anchor를 포함한다. 원 MAIN의 E, 동결 Chronos-Bolt-small FP32, D, 현재 상태 C개 우회 및 저장 mean/std를 그대로 복원한다. RAW는 같은 F와 원 전처리를 사용한다.

Robin C17/K4, Beijing C11/K2와 seed92601·92602를 각각 별도 프로세스에서 실행한다. 첫 1개·8개 반복 개발 창을 사용하고 RAW_BOLT/MAIN 모두에 동일한 두 전처리를 적용한다. baseline은 원 `experiment.preprocess`, 후보는 채널별 마지막 관측 위치를 누적 최대 인덱스로 찾아 gather하는 NumPy 구현이다. 창 시작의 FIT mean reset, 과거 관측 mask, FP32 정규화 연산식을 유지한다. 후보는 forecasting/decoder 학습을 추가하지 않는다.

`history_physical`은 fill 전의 원 물리 과거 창이어야 한다. 채워진 정규화 history를 역변환해서 전처리 시간을 재면 원 missing-data 작업량과 같다는 보장이 없으므로 실행 경로가 이를 대체 입력으로 만들지 않는다. 기존 bridge 입력 NPZ가 이 필드를 보존하도록 bridge 담당자가 작업 중이다. 실제 입력에서 원 loop 전처리와 stored history, 벡터 후보가 모두 byte-exact인지 먼저 검사한다.

## 계측과 출력

각 조건은 warmup2/repeat7 전체 경로를 계측한다. CPU causal fill·정규화, H2D, E/F/D, anchor, inverse, 물리 출력 D2H, 마지막 synchronize를 포함한다. 디스크·모델/CP load, reference 추론, normalized 출력 복사, 해시와 파일 저장은 측정 구간 밖이다. normalized tensor의 CPU 복사는 timer 뒤에서 별도로 수행한다.

그 뒤 별도 warmup2/repeat7에서 fill, normalization, H2D, E, F, D, anchor/inverse, D2H를 단계마다 GPU synchronize해 측정한다. D는 `D(z_F)`와 `D(z_last)` 모두를 포함한다. RAW의 E/D는 host identity bookkeeping이며 실제 encoder/decoder kernel 비용으로 해석하지 않는다. 동기화가 추가되므로 단계 시간의 합을 전체 경로의 원래 latency로 보고하지 않는다. 같은 장비·프로세스의 짧은 반복이며 독립적인 장비 반복 실험이나 유의성을 주장하지 않는다.

전체 및 분해 경로의 **실제 7회 normalized/physical 출력**을 모두 NPZ로 저장한다. 독립적으로 원 `engine.predict`와 원 `controls.deployment`를 같은 GPU batch로 호출한 reference를 함께 저장한다. matching-batch normalized bound는 `atol1e-5/rtol2e-6`, physical bound는 원 `atol.002/rtol2e-5`를 유지한다. 원 bulk science 비교는 별도 필드에 남기며 batch 차이에서 실패한 값을 새 기준 통과로 고쳐 쓰지 않는다. 맞춘 batch의 parity 실패는 해당 실행을 FAILED_PRESERVED로 종료한다.

동결 F의 old/Bi 두 해시 알고리즘, MAIN codec/CP, 원 source, 실제 실행 helper source 및 bridge의 SHA를 전후 확인한다. helper source copy와 각 route RESULT, 전체 terminal RESULT를 새 전용 폴더에 저장한다. 원 결과를 덮지 않는다. `followup_runtime.FollowupLimit(tag,gpu=True)`로 현재 캠페인의 남은 GPU·CPU·wall·storage를 공유하고 원 old Limit를 재시작하지 않는다. 로컬 watchdog600초와 cooperative check를 사용한다. `COMPLETE`는 요청한 한 dataset/seed의 8 route 완료만 뜻하며 전체 연구 완료가 아니다.

## 실제 실행 전 의존 조건

- Bi 완료 audit/report/manifest SHA와 실제 input bridge COMPLETE가 필요하다.
- `followup_runtime.py`와 원 physical 창을 저장하는 bridge artifact가 필요하다.
- 루트 담당자가 실제 CLI를 승인하기 전 이 준비 작업에서는 scientific NPZ/CP/model/GPU를 실행하지 않는다.

승인 뒤 root가 새 tag로 사용하는 예시는 다음과 같다. `BRIDGE_RESULT`는 실제 생성된 COMPLETE 결과의 경로로 바꿔야 한다.

```powershell
.venv/Scripts/python.exe research/smallk_failure_followup_20261014/profile_latency.py --describe
.venv/Scripts/python.exe research/smallk_failure_followup_20261014/profile_latency.py --execute-after-bi --bridge-result BRIDGE_RESULT --dataset robin --seed 92601 --tag robin92601_latency_v1
.venv/Scripts/python.exe research/smallk_failure_followup_20261014/profile_latency.py --execute-after-bi --bridge-result BRIDGE_RESULT --dataset robin --seed 92602 --tag robin92602_latency_v1
.venv/Scripts/python.exe research/smallk_failure_followup_20261014/profile_latency.py --execute-after-bi --bridge-result BRIDGE_RESULT --dataset beijing_aotizhongxin --seed 92601 --tag beijing92601_latency_v1
.venv/Scripts/python.exe research/smallk_failure_followup_20261014/profile_latency.py --execute-after-bi --bridge-result BRIDGE_RESULT --dataset beijing_aotizhongxin --seed 92602 --tag beijing92602_latency_v1
```

## 준비 검증

`test_profile_latency.py` **17 synthetic CPU tests PASS**. missing first/all-missing 채널과 이후 시점 perturbation, 원 loop/벡터 causal fill의 정확 일치, anchor 연산 순서, 전체/분해 출력 동일성, 실제 반복 출력 저장, bulk 실패와 matching-batch 통과의 분리, parity 실패 보존, cooperative budget 중단, 실패 조건의 speedup 집계 제외, opt-in/무자산 describe를 확인했다. resource constructor가 예산 때문에 거절돼도 FAILED_PRESERVED RESULT가 남는지와 보호 TEST hook이 bridge scientific read보다 먼저 설치되는지도 검사했다. 처음 반복-loop 입력 shadowing 오류를 발견했고 실제 실행 전 고쳤다. 이 검사 통과는 실제 forecast 개선이나 실제 latency 개선의 증거가 아니다.

파일: [profile_latency.py](profile_latency.py), [test_profile_latency.py](test_profile_latency.py).
