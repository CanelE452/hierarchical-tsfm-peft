# Electricity K81: 기존 두 번째 학습률의 제한된 재검사

이 모듈은 MASTER §7.3의 고정 두 학습률 `1e-3`, `1e-4` 중 두 번째 값을 실행하는 후속이다. 새 모델이나 새 학습 목표를 제안하지 않는다. 현재 K81의 B0/B2는 `1e-3`, 최대 .25 epoch의 A_stop 선택에서 CP0를 골랐다. 기존 학습률의 학습량만 늘려서는 최적화 조건과 구조적 한계를 충분히 나누기 어려워, 같은 seed·초기화·데이터 조건에서 작은 학습률을 확인한다.

현재 구현·허용 범위는 Electricity C321 / L512 / H48 / primary / seed92601 / K81의 B0와 B2 두 조건뿐이다. 다른 데이터·family·K·seed·학습률을 자동 추가하지 않는다.

```text
parent                                             new run
primary_electricity_E1_B0_k81_s92601                parent + __lr0001_e1
primary_electricity_E1_B2_k81_s92601                parent + __lr0001_e1
```

## 비교 계약과 실제 실행

`SOURCE_CONTRACT.json`과 실제 완료 parent receipt, 선택 checkpoint 파일 SHA, A50/A60 분할과 scaler, 여섯 과학 코드 SHA, 실제 데이터 SHA, 로컬 small checkpoint 파일 존재·SHA를 GPU 사용 전에 확인한다. 허용된 parent가 CP0를 선택했다는 조건도 검증한다. 원 spec에서 바뀌는 항목은 **run_id, lr, epochs 세 개뿐**이다. 새 학습률은 `1e-4`, epoch 목록은 `[.25, 1]`로 고정한다. stage label을 포함한 다른 spec 필드는 그대로 복사한다.

새 ID와 등록 확장 receipt를 중앙 OS ownership lease를 얻은 직후, 실제 모델 로드·학습·V 수치 생성 전에 기록한다. 따라서 기존 GPU owner와 registry를 동시에 수정하지 않는다. 원 parent의 선택 weights/optimizer/RNG를 이어 쓰지 않는다. 동일 seed로 A50 search를 새로 시작하고 A_stop 정규화 채널 macro MSE로 CP0/.25/1 중 선택한 뒤, 새 A60 scaler·PCA·초기화에서 선택 epoch를 refit한다. 새 search/refit의 초기화 SHA와 scaler를 원 parent의 실제 stage receipt와 다시 대조한다. 같은 decoder **구조와 초기화 조건**을 의미하며 학습 뒤 decoder 파라미터 수치가 같다는 뜻은 아니다.

성공한 새 fit은 같은 중앙 lease 안에서 `ParentPool.predict(new_id)`를 호출한다. 신규 V cache가 없던 조건은 frozen F stream이 실제 증가해야 하며 cache-only 반환을 거부한다. 예측 metadata의 새 run ID·선택 artifact SHA·data SHA와 실제 F counter/ledger를 대조한다. V는 새 fit의 실제 추론·기록에 사용하며 **학습률이나 checkpoint 선택에 사용하지 않는다**. 부모 M/E2, 기존 V 수치와 원 결과 파일은 교체하지 않는다.

이미 완료된 동일 새 ID를 다시 실행할 때만 exact contract/SHA가 맞는 fit·V cache 재사용을 허용하고 `fit_artifact_reused`, `V_prediction_executed_now`, `actual_new_V_F_streams`를 분리 기록한다. 새 run의 중단 carry는 같은 명령·같은 contract로 재개할 수 있다. 원 parent의 과거 weights를 새 연구의 optimizer 상태로 재활용하지 않는다.

## 자원·상태와 한계

기존 캠페인의 GPU process-wall 누계 43,200초, wall 48시간을 유지하고 **9,000 GPU process 초를 남기는** 중앙 `GPUReservation`을 사용한다. 두 조건은 한 lease와 lazy small F를 공유한다. 이미 GPU owner가 있으면 기다리는 queue를 만들지 않고 DEFERRED로 반환한다. 다른 프로세스를 종료하지 않는다. 한 조건의 구현/수치 오류는 기록하고 나머지 조건을 계속한다. 예산이 끝나면 current fit/carry를 보존하고 RUNNING 상태를 남기지 않는다.

완료 fit과 실제 V가 있어도 최대 1 epoch이므로 `PARTIAL_SCREEN_LT3`와 원 core의 `LIMITED_SCREEN`을 남긴다. CP0 재선택, 마지막 checkpoint 최적, 학습 곡선의 추가 하강 여부를 관찰해야 한다. 개선이 없다고 수렴 후 방법의 한계가 입증되는 것은 아니다.

원 LR의 현재 최대 .25 epoch과 새 LR의 최대 1 epoch에서 선택한 V 차이는 **학습률 단독 효과가 아니다**. 공통 .25 epoch의 A_stop 학습 경로, 그리고 이후 동일 최대 epoch의 두 LR 결과를 구분해 판단해야 한다. V를 보고 더 좋은 LR를 고르거나 원 M/E2 부모를 자동 교체하지 않는다. 최종 decoder·결합 출력이 조건부 중앙값이라는 보장도 추가하지 않는다.

선택 lock이 이미 동결됐다면 등록·GPU 사용 전에 거부한다. 이때는 기존 T로 조건을 다시 고르지 않고 새 campaign/별도 post-T 개발을 분리해야 한다. TEST 사용, 큰 backbone 재학습, 설치, commit/push는 수행하지 않는다.

## 실행과 이후 남는 항목

환경 변수 `INFORMS_ASSET_ROOT`, `INFORMS_CACHE_ROOT`, 검증된 `$py`를 설정하고 이 연구 디렉터리에서 실행한다. 활성 기존 GPU 작업이 끝난 다음 호출한다.

```powershell
& $py -X utf8 -u lr_followup.py --parent-id primary_electricity_E1_B0_k81_s92601 --parent-id primary_electricity_E1_B2_k81_s92601 --epochs .25 1
```

새 LR의 epoch3 연장은 기존 `deepen.py`로 별도 사전 등록한다.

```powershell
& $py -X utf8 -u deepen.py --parent-id primary_electricity_E1_B0_k81_s92601__lr0001_e1 --max-epoch 3
& $py -X utf8 -u deepen.py --parent-id primary_electricity_E1_B2_k81_s92601__lr0001_e1 --max-epoch 3
```

원 LR과 epoch를 맞출 후속 명령은 다음과 같다. 실제 잔여 자원과 다른 실행 순서를 확인한 뒤 사용하며 이 문서가 자동 실행하지 않는다.

```powershell
& $py -X utf8 -u deepen.py --parent-id primary_electricity_E1_B0_k81_s92601 --max-epoch 1
& $py -X utf8 -u deepen.py --parent-id primary_electricity_E1_B2_k81_s92601 --max-epoch 1
```

미실행으로 남는 범위는 B4 K81의 두 번째 LR, B0/B2/B4의 K161·K321 등 다른 K, 다른 데이터, 추가 seed, 동일 예산의 LR 대사, 독립 재검산·실제 비용·checkpoint replay·최종 선택/T다. 나머지 승인된 grid는 조건별 준비·사전 등록·잔여 자원 확인 후 진행해야 한다. 현재 runner는 이 항목을 다른 모델로 대신하거나 완료로 표시하지 않는다.

읽기 전용 실제 preflight에서는 두 부모와 데이터/모델 자산 조건이 맞고 새 config SHA가 서로 구분됨을 확인했다. 이것은 학습·V 추론 실행이나 성능 개선의 증거가 아니다. `test_lr_followup.py`는 synthetic 임시 자산과 mocked fit/F만 쓰는 CPU runner 검사이며 실제 과학 결과가 아니다.

```powershell
& $py -X utf8 -m unittest test_lr_followup -v
```
