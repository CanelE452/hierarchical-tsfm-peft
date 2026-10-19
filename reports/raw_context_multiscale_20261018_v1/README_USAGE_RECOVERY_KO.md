# 실제 완료 검산과 복구 실행 명령

원 본체 실행 `TERMINAL_CORE_002.json`은 학습·56개 V 예측·120개 whole-path 비용 측정을 끝낸 뒤 검산기의 메타데이터 검사에서 실패했다. 이 FAILED 파일과 원래 `VALIDATION_REPORT.json`의 FAIL은 보존한다. 첫 CPU 복구도 별도 FAILED로 남아 있다. 학습·예측·비용 결과를 다시 만들거나 점수·선택·허용오차를 바꾸지 않고, 기존 생성기가 사용한 계약을 검사하는 CPU 복구 002가 별도 성공 terminal과 PASS 검산 파일을 만든다.

두 수선은 정확히 등록한 Direct G 두 artifact의 `NO_FOUNDATION` 표기와 기존 Jena LoRA·PeMS32 Direct G 각각의 config 해시 생성 공식에 한정한다. 모델 state, scaler, data, source, canonical target/mask/origins, FP64 재채점, 배포 parity, 비용 원 trial 검사는 유지한다. 구체적인 source·실패·원 실행 예산·새 실행기 지문은 `COMPLETION_RECOVERY_REGISTRATION_001.json`과 `COMPLETION_RECOVERY_REGISTRATION_002.json`에 연결돼 있다.

## 환경과 읽기 전용 확인

저장소 최상위에서 기존 PyTorch/CUDA 환경의 Python을 사용한다. 아래 `<asset-root>`는 원래 private data/checkpoint를 보관한 checkout의 root로 바꾼다. 새 빈 폴더를 자산 root로 사용하지 않는다. 공개 저장소에는 원자료·가중치·개별 예측 배열을 올리지 않으므로, 공개 결과의 표·그림 재생성과 private 자산을 요구하는 모델 검산을 구분한다.

실제 성공한 복구 receipt와 원 실패·등록·source·산출물 지문을 다시 확인하는 읽기 전용 명령은 다음과 같다.

```text
python -c "from research.raw_context_multiscale_20261018_v1 import common; from research.raw_context_multiscale_completion_recovery_20261019_v1.revision2 import validate_completion; print(validate_completion(common.paths()))"
```

정상 종료는 `TERMINAL_COMPLETION_RECOVERY_002.json`의 COMPLETE/exit0와 `RECOVERY_VALIDATION_REPORT_002.json`의 PASS를 요구한다. 원 실패 파일을 COMPLETE/PASS로 바꾸지 않는다. 원래 실행기의 `verify`는 과거 형식 검사 실패를 그대로 재현할 수 있으므로, 이미 완료한 캠페인의 확인에는 위 복구 검증 진입점을 사용한다.

## 실제 실행 단계

아래는 실제 구현한 진입점이다. 등록과 완료는 새 002 출력이 없을 때만 사용하며, 기존 등록·결과가 있으면 보존을 위해 재실행을 거절한다. private `driver_completion_recovery_002.py`가 실제 CPU worker 종료코드 0을 확인한 뒤에만 성공 terminal을 작성한다. `complete` 명령 하나만 수동 실행했다고 terminal 성공을 선언하지 않는다.

```text
python -m research.raw_context_multiscale_completion_recovery_20261019_v1.revision2 register --asset-root "<asset-root>"
python -m research.raw_context_multiscale_completion_recovery_20261019_v1.revision2 complete --asset-root "<asset-root>"
```

후속 단계는 위 성공 lineage를 먼저 검사하고 각 단계의 기존 과학적 검사를 그대로 수행한다. CP0는 고정된 학습 전 모델의 A_stop 평가 10개이며, 새 학습이나 재선택이 아니다. final은 별도 새 `final/` 폴더만 생성한다. 독립 검산은 저장 FP32 배열과 원 비용 trial을 읽는 CPU 작업이다. M_cal은 고정된 최종 40개 결합 모델의 전체 M_cal 적합 오차를 CPU에서 평가한다.

```text
python -m research.raw_context_multiscale_completion_recovery_20261019_v1.revision2 cp0 --asset-root "<asset-root>"
python -m research.raw_context_multiscale_completion_recovery_20261019_v1.revision2 final --asset-root "<asset-root>"
python -m research.raw_context_multiscale_completion_recovery_20261019_v1.revision2 independent --asset-root "<asset-root>"
python -m research.raw_context_multiscale_completion_recovery_20261019_v1.revision2 mcal --asset-root "<asset-root>"
```

Head 자체 비용 16개는 모델을 다시 적합하지 않고 고정 head 8개×B1/B8을 fresh CUDA process에서 측정한다. 전체 경로 120개 비용에 더하거나 부모 latency를 빼서 해석하지 않는다. 측정 구간은 GPU에 준비된 M_cal 입력·부모 예측에서 head의 feature 구성과 forward, GPU 출력 동기화까지다.

```text
python -m research.raw_context_multiscale_head_cost_20261019_v1.run collect --asset-root "<asset-root>" --scope reports/raw_context_multiscale_20261018_v1/HEAD_COMPONENT_COST_SCOPE_RECOVERY_002.json --preflight reports/raw_context_multiscale_20261018_v1/HEAD_COMPONENT_COST_PREFLIGHT_RECOVERY_002.json
```

## 공개 수치만으로 재생성

각 출력 폴더는 존재하지 않거나 해당 명령이 허용하는 새 빈 private cache 폴더여야 한다. 기존 원본 보고·표·그림을 출력 위치로 지정하지 않는다. final 재생성에는 공개 CSV/JSON과 등록된 생성 코드만 사용한다.

```text
python -m research.raw_context_multiscale_supplement_20261019_v1.report replay --source-dir reports/raw_context_multiscale_20261018_v1/final --output-dir .cache/raw_context_multiscale_20261018_v1/final_public_replay_new
python reports/raw_context_multiscale_20261018_v1/reproduce_mcal_means.py --source-dir reports/raw_context_multiscale_20261018_v1 --output-dir .cache/raw_context_multiscale_20261018_v1/mcal_mean_public_replay_new
python -m research.raw_context_multiscale_head_cost_20261019_v1.run replay --source-dir reports/raw_context_multiscale_20261018_v1 --output-dir .cache/raw_context_multiscale_20261018_v1/head_component_public_replay_new
```

성공 선언에는 실제 출력 파일별 원본 바이트 비교가 필요하다. 명령의 종료코드나 파일 개수만 같다는 것은 같은 표·그림의 증명이 아니다. M_cal 평균 20행은 두 seed의 지정된 8개 metric만 산술평균하며, 선택 epoch·계수·지문 등의 보조값은 원 seed별 기록으로 유지한다.

## 예산·재개 한계

모든 실제 복구·CP0·M_cal·head 진입점은 원 실행 시작과 6시간 wall/4시간 GPU 예약 상한을 유지한다. 원 deadline은 2026-10-19 17:12:28 KST이며, 이 문서나 새 Python process가 예산을 초기화하지 않는다. 실패·상한 도달·이미 있는 출력은 보존한다. 과거 실패 명령의 무조건 재시도나 봉인 파일 덮어쓰기는 재개 절차가 아니다.

원래 `audit/smoke/execute/verify/report` 인터페이스와 초기 실제 실행 근거는 `final/CLI_USAGE_KO.md`, 원 계획과 preflight에 남아 있다. 이번 완료 검산·후속 진단의 실제 복구 경로는 이 문서의 002 진입점이다. deadline 이후 추가 실험이나 새 시도는 새로 승인·등록해야 하며, 미수렴 cap 결과를 수렴 또는 구조적 실패로 바꾸어 표현하지 않는다.
