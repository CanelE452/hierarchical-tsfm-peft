# Beijing 필수 최소 비교의 새 CPU 실행 경로 — 준비 상태

현재는 [실행 소스](required_minima_runner.py)와 [stdlib 합성 테스트](test_required_minima_runner.py)를 작성한 단계다. 이 구현 담당자는 실제 Task·모델·배열·학습·추론·자식 프로세스와 `--seal-plan`을 실행하지 않았다. 정밀 비용 측정과 겹치지 않도록 테스트 실행도 ROOT의 quiet-window 확인 이후로 미뤘다. **테스트 통과, 실제 probe 성공, 학습 완료 또는 필수 비교 완료를 주장하지 않는다.**

이전 미봉인 runner `98675ac4…`와 당시 테스트·README는 수정 전 private snapshot에 동일 bytes로 보존했다. 이 구현 담당자는 이전 준비본의 metadata/proposal·테스트·과학 실행을 수행하지 않았다. 이어진 `0119bec3…` 준비본도 수정 전 별도 private snapshot에 보존했다. ROOT는 그 준비본의 stdlib 테스트11개(0.093초)와 실제 metadata/proposal 검사가 통과했다고 보고했으며, PLAN 봉인·실제 probe·fit은 수행하지 않았다고 확인했다. 아래 F 모듈 전용 backward 계측과 실제 작은 Torch 회귀는 새 소스 준비 상태이며 실행 검증 전이다.

원 MASTER의 미완료 최소 비교를 원 방법 그대로 CPU에서 보강한다. T를 관찰한 뒤 수행하는 V-only 개발이므로 새로운 독립 일반화 검증으로 부르지 않는다. 원 GPU label의 부족분이나 원 registry 상태를 자동으로 완료로 바꾸지 않는다.

## 정확한 여섯 조건

K=6,11 각각 B2, LAG_CAUSAL, LAG_BIDIR를 실행한다. 원 ID는 `primary_beijing_E1_B2_k{K}_s92601`, `primary_beijing_E3_LAG_CAUSAL_k{K}_s92601`, `primary_beijing_E3_LAG_BIDIR_k{K}_s92601`다. 새 ID는 각각 `postT_cpu_beijing_E1_B2_k{K}_s92601_required_v1`, `postT_cpu_beijing_E3_LAG_CAUSAL_k{K}_s92601_required_v1`, `postT_cpu_beijing_E3_LAG_BIDIR_k{K}_s92601_required_v1`다.

원 registry의 `MANDATORY_MINIMUM`·`planned_epoch_endpoints=0.25|1|3`와 여섯 원 `SOURCE_CONTRACT.json`의 실제 0.25 screen recipe를 확인한다. 소스·손실·encoder 구조·분할을 바꾸지 않는다. 원 recipe에서 바뀌는 것은 새 run ID, CPU device, 이미 등록된 전체 0.25→1→3 search 일정과 명시적 post-T provenance뿐이다. 기존 run/checkpoint를 이어 학습하지 않는다.

원 `fit_base.create_model`은 세 family와 K6/K11을 지원한다. K6은 원 A-fitted PCA 초기화, K=C=11은 원 identity 초기화를 유지한다. B2는 원 pointwise encoder와 B2 front, LAG_CAUSAL/LAG_BIDIR는 원 B0 front와 causal7/bidirectional7 encoder다. 새 encoder나 목적 함수를 만들지 않는다. bidirectional7도 주어진 과거 창 안의 위치를 사용하며, 새 미래 입력을 추가하지 않는다.

## 봉인과 데이터 계약

기본 명령은 stdlib metadata 출력만 하며 과학 모듈을 수입하지 않는다.

```powershell
python -B required_minima_runner.py
```

`R`은 원 campaign, `C`는 원 private campaign cache, `A`는 기존 자산 저장소다. 절대경로를 소스에 넣지 않고 ROOT가 CLI 인자로 제공한다. `--seal-plan`은 새 private 하위 디렉터리에 `PLAN.json`, `REGISTRY.csv`, `SEAL_RECEIPT.json`만 만든다. 공개 PLAN/REG 교체나 과학 실행을 하지 않는다.

```powershell
python -B required_minima_runner.py --public-root <R> --cache-root <C> --asset-root <A> --seal-plan --output <C의_새_private_후보폴더>
```

ROOT는 후보와 세 소스 파일을 검토한 뒤 새 namespace에 PLAN/REG를 명시적으로 봉인하고 실제 PLAN SHA를 실행에 전달한다. PLAN의 schema는 `POST_T_REQUIRED_MINIMA_CPU_PLAN_v1`이며 핵심 필드는 다음과 같다.

- `source_reference`: 원 PCA PLAN의 경로·고정 SHA.
- `source39`, `immutable_public`: 원 과학 소스 39개와 registry·selection·score·cost·T 표 9개 SHA.
- `upstream_recipe_metadata`: 원 여섯 `SOURCE_CONTRACT.json`의 SHA.
- `task_contract`: PCA PLAN의 Beijing 원 계약 전체와 search/refit scaler SHA. L512/H48/C11/V43, A_fit 끝10519, A_stop/refit 끝12622를 유지한다.
- `conditions`: 새 여섯 ID, 원 ID, 원 campaign recipe, CPU spec.
- `runner_sha256`, `postT_registry_sha256`, model revision/foundation SHA, 실제 normalization source SHA.
- CPU2, 각 RSS3GiB, 외부 최대2worker·총6GiB, C+R30GiB와256MiB 여유, 원 deadline1792222550.

source39는 39개 Python 파일이라는 뜻이 아니다. 원 목록의 JSON/MD도 포함한 정확한 집합을 재사용한다. 새 PLAN은 원 T selection lock을 계속 잠근 상태로 원 입력의 SHA를 전후 검사한다.

## 조건별 실제 probe 후 새 fit

각 fit 전에 **같은 새 조건**의 실제1~3 effective update probe가 필요하다. 원 full A50 legal origins9960과 loss normalization을 사용하고 micro4/effective16, seed92601, AdamW lr0.001/weight_decay0을 유지한다. 임의의 작은 학습 모집단으로 바꾸지 않는다.

probe는 실제 F logical/physical streams=`16 × updates × K`, row/physical calls=`4 × updates`를 검사한다. forward의 `requires_grad` 관측과 실제 F 경유 backward를 구분한다. 디코더도 같은 잠재 `z`를 직접 쓰므로 `z` tensor hook만으로는 F 경유 gradient를 증명할 수 없다. probe에서 frozen foundation **모듈**에 `register_full_backward_hook`를 달고 `grad_input[0]`만 기록한다. 이 이벤트는 F 모듈 경계 안을 통과한 입력 gradient의 `[4,512,K]` shape·finite 여부·absmax이며, 디코더의 직접 `z` 분기는 포함하지 않는다. 정확히 `4 × updates`개의 실제 이벤트, 모두 finite, probe 전체에서 적어도 하나의 nonzero gradient가 필요하다. `actual_backward_input_gradients_observed`·count·finite·nonzero count·absmax와 `actual_backward_input_gradient_scope=F_module_grad_input_only`, 원 이벤트를 기록하고 fit의 probe proof gate에서도 같은 조건을 재검증한다. hook은 `None`을 반환하여 gradient를 수정하지 않으며 probe 뒤 forward/backward handle과 상태를 폐기한다. 실제 fit에는 이 hook을 넣지 않는다. 학습 parameter의 finite gradient, F parameter gradient 없음, F weight SHA 불변은 별도로 확인한다. probe 상태·optimizer를 저장하거나 fit에 재사용하지 않는다. probe 성공은 CPU 완주·메모리·수렴 보장이 아니다.

테스트의 실제 작은 Torch 회귀는 `INFORMS_REQUIRED_AUTOGRAD_TOY=1`로 ROOT가 비용 측정 종료 후 명시 opt-in할 때만 실행된다. 같은 `z`를 F와 디코더 직접 분기가 함께 쓰는 fixture에서 F 출력을 detach하면 `z.grad`는 존재해도 F 모듈 backward 이벤트가 없어 거부해야 한다. detach하지 않으면4개 실제 모듈 gradient 이벤트가 있어야 한다. 기본 테스트는 이 회귀를 skip하며 Torch를 수입하지 않는다. 이는 미봉인 검사 경로의 false-positive 방지 회귀이며 기존 실제 실험에서 오류가 관측되었다는 뜻이 아니다.

```powershell
python -B required_minima_runner.py --public-root <R> --cache-root <C> --asset-root <A> --plan-sha <봉인_PLAN_SHA> --mode probe --run-id <여섯_ID중_하나> --probe-updates 3 --deadline <원상한_이내> --output <새_public_diagnostics폴더> --execute
python -B required_minima_runner.py --public-root <R> --cache-root <C> --asset-root <A> --plan-sha <봉인_PLAN_SHA> --mode fit --run-id <같은_ID> --probe-result <실제_probe_RESULT.json> --probe-result-sha <실제_SHA> --deadline <원상한_이내> --output <다른_새_public_diagnostics폴더> --execute
```

원 `fit_base.fit_base`를 직접 호출한다. 처음부터 search0/0.25/1/3을 마치고 A_stop으로 epoch를 선택한 뒤 새 A60 model로 refit한다. selected epoch0이면 refit update0을 그대로 보존한다. 전체 search의 미완료를 3epoch 완료로 바꾸지 않는다. 원 `training_status`와 `selected_epoch`를 그대로 기록하며 수렴을 단정하지 않는다.

완료 parent에만 원 `prediction_cache.predict_parent`를 호출하여 V43을 생성한다. F stream은 K6이면258, K11이면473이어야 한다. 다른 split과 cached V 재사용은 거부한다. 원 BASE_FIT_RECEIPT·stage RECEIPT·CURVE·selected artifact·V manifest/statistic은 원 API가 새 namespace에 기록한다. 새 partial/claim도 재개하거나 덮어쓰지 않는다.

## 실행 후 남는 검증과 한계

`COMPLETE_FRESH_CPU_FIT_AND_FULL_V_PENDING_INDEPENDENT_VALIDATION`은 독립 검산 전 상태다. ROOT는 원 `audit_predictions.audit_prediction_bundle`로 full V43을 재집계하고 원 `route_predictions.replay_route`로 고정6 origin의 실제 CPU checkpoint replay를 수행해야 한다. 새 registry의 exact byte alias가 필요한 경우 새 namespace 안에서만 준비한다. 현재 runner는 이 별도 검산을 실행하지 않는다. 후속 검산 담당은 새 PLAN·여섯 ID·parent/scaler/source/array/artifact binding을 그대로 연결해야 한다.

E4의 B0 K6/K11도 완료·동일 데이터/scaler/recipe·검산 조건이 실제 확인된 경우에만 참조한다. 지금은 완료를 가정하지 않는다. 원 기존 GPU/CUDA 비용과 새 CPU 비용을 섞어 속도 우위를 주장하지 않는다.

ROOT는 기존 비용 큐가 모두 종료된 뒤 외부 CPU process-tree supervisor로 실행해야 한다. runner의 자체 RSS/deadline watchdog은 외부 PID/create_time·자식 closure 증거를 대신하지 않는다. 보호된 T 평가, 원 선택 해제, 원 결과 수정, 자동 commit/push는 구현하지 않는다. 원 safeFIT loader가 파일 내부의 T bytes를 물리적으로 역직렬화할 수 있는 사실과 T target/scoring을 요청하지 않는 계산 범위를 구분한다.
