# G A+M 대조군의 학습 길이 보완

## 목적과 범위

E2의 H는 이미 0.25·1·3 epoch를 실행했지만 원 `G_REFIT_APLUSM` 통합 실행은 0.25 epoch다. 짧게 학습한 G와 H를 비교하면 G의 학습 부족을 방법 차이와 혼동할 수 있다. `control_deepen.py`는 실제 완료한 원 G만 읽고, 별도 `원 ID__e3`에 같은 초기화·seed·LR·모델·손실·배치·데이터 조건으로 **새로** 0.25·1·3 epoch를 학습한다. LoRA 확장은 이 실행기에 포함하지 않는다.

원 G의 가중치·optimizer·RNG·결과를 이어받지 않는다. 기존 `fit_control.py`와 과학적 핵심 6개 파일도 수정하지 않는다. 원 0.25 결과를 보존하는 fresh 비교다. 동일 seed라는 선언에 더해, 새 fit receipt의 실제 초기 모델 상태 SHA가 원 G의 SHA와 같은지 검사한다.

## 데이터와 선택 계약

- 정규화 통계는 A_fit+A_stop의 A60, 학습 표적의 마지막 경계는 A+M의 A70이다. 원 G와 동일하다.
- 체크포인트는 기존 G와 동일하게 V_select의 정규화 채널 macro MSE로 고른다. 따라서 V 수치는 선택 집합 증거이며, 독립 일반화 검증이라고 해석하지 않는다.
- 새 체크포인트로 V 추론을 실제 실행하고 저장한다. 신규 예측 manifest가 없는데 캐시 재사용이라고 응답하면 거부한다. 이후 같은 완료 artifact·contract의 재호출은 명시적으로 fit/cache 재사용으로 기록한다.
- 이 실행기는 foundation 객체를 만들거나 전달하지 않는다. source contract의 `NO_FOUNDATION`, 학습 receipt의 `F_logical_streams=0`, 실제 추론 ledger의 expected/actual F streams=0을 확인한다. 공통 추론 helper의 `F_calls=1` 필드는 논리적인 경로 multiplicity이므로 실제 foundation 호출 수로 해석하지 않는다.
- 원 CONTROL_SOURCE_CONTRACT의 실제 spec·7개 소스 SHA·data SHA·config SHA, private run contract, A60 scaler, 학습/정규화 경계, 완료 artifact SHA를 모두 확인한다. preflight에서 모델 가중치를 로드하지 않는다.
- T_final은 접근하지 않는다. 최종 선택이 잠겼으면 preflight 단계부터 학습·등록을 거부한다.

## 자원과 실패 처리

원 GPU process 총량 43,200초, training 잔여 보존 9,000초, 원 캠페인의 48시간 종료 시각을 그대로 적용한다. OS GPU lease 안에서 새 ID를 불변 extension으로 등록한 **후** Torch import·모델 생성·학습 표적 생성으로 진행한다. 저장소 여유를 단계마다 확인한다. 실행 중 다른 소유자가 있으면 기다리는 프로세스를 만들지 않고 자원 대기로 기록한다.

데이터셋별 부모 누락·손상·학습 실패는 해당 경로에만 적용하며, 독립적으로 준비된 다른 데이터셋을 진행한다. 부분 target을 자동으로 덮거나 재개하지 않는다. 같은 새 `__e3` 계약의 부분 실행만 `--resume`으로 명시 재개할 수 있다. 원 0.25 부모는 재개하지 않는다. 이미 완성된 target도 source/config/data/초기화/artifact SHA가 모두 맞는 경우에만 재사용한다.

`COMPLETE_VALID`는 파일 생성과 실행 계약 완료 상태다. 마지막 체크포인트가 최선이면 `UNDERTRAINED_OR_BUDGET_LIMIT`, 그 외에도 `SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED`를 유지한다. 3 epoch 완료, 음성 결과, CP0 선택만으로 수렴이나 방법 한계를 선언하지 않는다. 독립 수치 감사와 보고서 통합은 별도 단계다.

## 해석에서 남는 차이

G는 A+M 전체를 3 epoch 학습하고 H는 M 구간을 3 epoch 학습하므로 같은 epoch 수라도 전체 update·표적 노출·고유 origin 수가 다르다. 이 보완은 짧은 G 학습이라는 반례를 줄이지만 동일 update 예산 실험은 아니다. 각 receipt의 updates, exposures, actual_epochs, train_origins, observed_target_exposures와 시간·메모리 비용을 함께 비교해야 한다. V 선택을 여러 epoch에서 수행한 점도 공통 선택 조건으로 명시한다.

## 실행

PowerShell에서 저장소 위치와 데이터/캐시 위치는 환경 변수로 받는다. 실제 원 부모 4건이 완료된 뒤 실행한다.

```powershell
if (-not $env:INFORMS_WORKTREE -or -not $env:INFORMS_ASSET_ROOT -or -not $env:INFORMS_CACHE_ROOT) {
    throw 'INFORMS_WORKTREE, INFORMS_ASSET_ROOT, INFORMS_CACHE_ROOT를 실제 경로로 설정하세요.'
}
$campaignDir = Join-Path $env:INFORMS_WORKTREE 'research/informs_broad_validation_20261015_v1'
Set-Location -LiteralPath $campaignDir
python control_deepen.py --dataset all4 --preflight
python control_deepen.py --dataset all4
```

일부 데이터셋만 실행하거나 **새 target의** 동일 계약 부분 실행을 명시 재개하려면 다음과 같이 실행한다.

```powershell
python control_deepen.py --dataset beijing --dataset electricity
python control_deepen.py --dataset beijing --resume
python -m unittest test_control_deepen -v
```

실행 영수증은 `execution_batches/control_deepen_<time_ns>_<uuid>/`에 새로 생성한다. 새 run의 출처는 `runs/<new ID>/CONTROL_DEEPEN_PROVENANCE.json`, 실제 fit/추론 증거는 기존 CONTROL_FIT_RECEIPT와 PREDICTION_MANIFEST 형식으로 저장한다. 비공개 checkpoint와 traceback은 설정된 private cache에 둔다.

## 구현 검증 기록

[확인] 2026-10-15 구현 직후 CPU fixture 9건이 통과했다. 합성 240×3 과거 데이터에서 원 0.25 G fit과 새 3 epoch G fit, 실제 저장 checkpoint의 V 추론을 수행했다. 새 fit은 20 update·303 exposure·actual_epochs=3이었고 원 초기 모델 SHA·A60 통계·A70 학습 경계가 같았다. 이 값은 **합성 fixture 검증 값이며 실제 데이터의 연구 결과가 아니다**.

검사는 등록→fit 순서, 독립 데이터셋 실패 격리, 원 파일 보존, 완성 target 정확한 재사용, 부분 target의 `--resume` 필수, 손상 checkpoint 거부, locked T 이전 거부, 지원 범위 제한, 별도 Python 프로세스에서 Torch 없는 import를 확인했다. 과학적 핵심 6개 소스 SHA는 검사 전후 같았다.

[확인] 같은 시점의 실제 4데이터셋 읽기 전용 preflight에서는 ready=0이었다. Beijing·Jena·Electricity·PeMS08 모두 원 `CONTROL_FIT_RECEIPT.json`이 아직 없어 `PARENT_NOT_COMPLETE`로 기록했다. 실제 등록·GPU 학습·추론·기존 작업 재개는 이 구현 검증에서 수행하지 않았다. 통합 큐에서 원 G가 완료되면 위 preflight를 다시 실행한 뒤 준비된 경로만 실행한다.
