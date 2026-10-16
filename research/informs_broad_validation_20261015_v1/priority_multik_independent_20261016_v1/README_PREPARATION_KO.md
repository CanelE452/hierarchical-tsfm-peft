# E4 독립 K3/6/11 안전 실행 어댑터 준비

현재 상태는 **PREP_ONLY / DRAFT_NOT_EXECUTABLE**이다. 기존 등록 세 조건의 실행 경로를 준비했으며 실제 prefix 숫자 해석·Task·모델·체크포인트 텐서·학습·추론·비용·통계 재계산·큐 시작은 모두0이다. ROOT의 새 PLAN 봉인이나 실제 입장도0이다. 현재 진행 중인 N4와 기존 N7/N8 소스·계획·결과는 수정하지 않는다.

## 원 조건과 질문

원 `postT_V_only_cpu_multik_20261016_v1/PLAN.json`의 SHA256은 `2051573b8dee48200eb34e96e37e7af0d197442619ea1ba52a36a80ad4e27aec`이며 REGISTRY SHA256은 `7bb634cf469105c455d552ffedeadbfcb324c2803a95829891589c7fa10c5144`다. 이 둘에 등록된 정확한 ID만 허용한다.

```text
postT_cpu_beijing_E4_INDEPENDENT_B0_k3_s92601_v1
postT_cpu_beijing_E4_INDEPENDENT_B0_k6_s92601_v1
postT_cpu_beijing_E4_INDEPENDENT_B0_k11_s92601_v1
```

`role=INDEPENDENT_B0`는 등록상 역할이고 원 모델 family는 `B0`다. 각 spec에는 `k_grid`와 `family_lock_sha`가 없다. `k_grid=[K]`를 임의 추가하면 같은 원 조건이 아니다. 원 stage label·upstream provenance·lr.001·seed92601·micro4/effective16·row128·checkpoint_inputs=False를 보존한다. 새로운 K, 방법, 데이터 또는 목적함수를 제안하는 구현이 아니다.

N7의 중단된 shared/max 부모 두 개를 재개하는 준비와 달리, 이 세 조건은 서로 독립적인 fresh 학습이다. 기존 CP나 부모 bundle, optimizer, RNG를 읽거나 복사하지 않는다. 새 namespace에 같은 등록 ID를 연결하되 원 namespace의 결과·partial·claim이 있으면 실행을 거부한다. 현재 준비 중의 경로 감사에서는 세 조건의 원 runs/predictions/fit claim을 발견하지 않았다. 실제 입장 시 다시 검사한다.

## 중요한 초기화 차이

원 `multik_runner.py:87–94`와 PLAN의 독립 recipe, `fit_base.create_model:69–113`을 대조했다. K3/6은 해당 A 구간에서 적합한 centered PCA로 E/center를 초기화한다. K11=C는 **E=I, center=0**으로 초기화한다. search A50과 fresh refit A60은 각자의 원 scaler/PCA를 적합한다. 새 모델을 두 번 미리 만들어 보는 probe는 추가하지 않는다. 실제 `_stage`가 반환한 normalization의 초기 basis/center를 원 함수로 대조하고 fingerprint를 남기도록 준비했다.

공유/max-only K11 부모에는 grid가 있어 PCA 초기화가 적용된다. 따라서 independent K11과 shared/max K11의 차이를 공유 학습의 순수한 효과로 단정할 수 없다. 이 차이를 고치려고 원 recipe를 바꾸지 않았다. 학습 가능한 E·lag D·현재값 anchor를 쓰는 독립 B0와, N3의 fixed PCA basis inverse는 서로 다른 경로다.

## 데이터와 경계

```text
Beijing original task   C11 / L512 / H48 / original permitted 21038 rows
실제 숫자 변환 범위       0:16830 only
A50 search fit end      10519
A_stop                  10519:12622
A60 fresh refit end     12622
M_cal                   12622:14726 (새 target 학습0)
V_select                14726:16830 / 43 origins / observed20539 labels
보호 T_final             16830:21038 (새 숫자 해석·채점0)
```

원 source는 28051행의 C-order physical/mask/times/columns archive다. 기존 N3 `safe_task`와 P1 bounded primitive를 재사용한다. 원 full-task data/array hash는 `INHERITED_NOT_RECOMPUTED`로 표시하고 실제 materialized prefix hash를 따로 남긴다. A50 mean/std/count, A50/A60 scaler와 원 V target/mask/origin/channel 네 fingerprint를 각각 대조한다. 원 compressed asset 전체의 opaque SHA나 ZIP buffering은 tail byte를 읽을 수 있으므로 “tail I/O0”으로 쓰지 않는다. T의 수치 변환0과 구별한다.

`data.load_task`, `data.split_ranges(truncated_task)`, 기존 `fit_base.fit_base`, `prediction_cache` 또는 `route_catalog.build_route(truncated_task)`를 호출하지 않는다. truncation 길이로 split을 다시 계산하면 계약이 달라진다. 원 `_stage`와 `deployment.Route`에는 명시한 원 경계를 전달한다.

## 준비한 실행 경로

- `fit`: 원 `_stage`로 search0/.25/1/3·A_stop 선택, 선택 epoch로 fresh A60 refit, 선택 bundle 저장, 원 Route full constructor로 실제 physical 전체 V43 추론을 하도록 준비했다. search 완료와 선택 refit epoch, 충분 수렴을 구분한다.
- `rescore`: 명시된 private V archive의 array/SHA·public 충분통계·SCORES를 P1 독립 FP64 코드로 대조한다. 이 분기에는 Torch·F 모델이 없다.
- `replay`: 정확한 새 bundle의 원 CPU Route로 6개 origin을 4+2 batch로 실제 재추론하고 기존 `atol=1e-5, rtol=1e-4`를 적용하도록 준비했다.
- `cost`: 예측 NPZ를 읽지 않고 manifest·실제 성공한 rescore/replay의 outer RESULT/SHA를 결속한다. 조용한 창의 fresh120초 ROOT proof가 필요하다. 원 `measure_balanced` warm10/timed20/stage20로 B1 또는 B8의 전체 physical preprocessing→P/E/F/D→FP64 inverse→CPU 반환을 측정하도록 준비했다. 모델 loading과 whole trial은 구별한다. ROOT가 fresh guard child 하나씩 따로 실행해야 한다.

정확한 선택 spec/state/model/scaler/새 PLAN/source를 original `Route` constructor에서 대조하고, 종료 시 route.close와 frozen F 불변을 확인한다. fullV43의 F stream은 K3/6/11 각각129/258/473, row128 physical calls는11이다. CPU6은18/36/66 streams와2calls다. B8은24/48/88 streams와1call이다. 비용50trial의 전체 카운터도 확인한다. K6 상수를 다른 K에 재사용하지 않는다.

실패와 traceback은 새 private failure 경로에 보존한다. 기존 source39/immutable9·등록·raw opaque SHA·snapshot→blob path/size/SHA·normalization source는 실행 전후 대조한다. 모델 symlink가 asset root 안의 정확한 pinned blob을 가리키는 조건을 검사하며 symlink 자체를 모델 손상으로 취급하지 않는다.

## 실행 게이트와 원 자원

기본 CLI는 입력0/read0/write0/NumPy0/Torch0이다. `PLAN_DRAFT.json`의 `runner_sha256=null`, `ROOT_after_N4_report_and_main_admitted=false`는 유지한다. `preparation_runner_sha256`/`preparation_test_sha256`는 준비 파일 바인딩이고 실행 승인이 아니다.

ROOT가 N4 보고와 main 게시를 마친 뒤 원 목적·남은 wall/저장을 검토하고, 별도 `PLAN.json`을 `SEALED_BY_ROOT` 및 명시 양성 admission/current runner SHA로 작성한 경우에만 실행이 가능하다. 각 실행에는 `--execute --root-triggered`와 명시 public/cache/asset roots·sealed PLAN SHA·정확한 run ID·mode·live deadline·새 diagnostic output이 모두 필요하다. 새 큐나 연쇄 실행기는 포함하지 않았다.

```text
CPU threads             2 / GPU0
worker RSS              3GiB
동시 모델 worker         최대2, 합계6GiB (ROOT supervisor)
저장                    기존 public+private 합30GiB, reserve256MiB
원 worker deadline      1792222550 (연장/초기화하지 않음)
cost quiet              status=QUIET_CPU_COST_WINDOW,
                        active_model_workers=[], active_cost_workers=[],
                        captured_unix가현재에서120초이내
```

원 `_stage`의 optimizer-boundary/deadline CP만 보존한다. Ctrl+C 저장, 신호별 autosave, 이번 adapter의 partial resume는 구현하거나 약속하지 않는다. 원 stage가 `BUDGET_LIMIT_PRESERVED`를 반환하면 종료 직전 deadline 검사로 이를 FAILED로 덮지 않는다. 상한이나 원 계약을 못 지키면 해당 조건만 실제 미완료로 남겨야 한다.

## 검증 수준과 현재 한계

현재는 stdlib synthetic16개가 PASS다. 초기에는 구현 부재로14개 RED를 확인했고, 이후 JSON round-trip 검사에서 정수 metadata key가 문자열로 바뀌는 불일치를 실제로 잡아 수정했다. 저장된 partial 뒤의 final deadline에 대한 fake execute 경로도 검사했다. 테스트는 임시 작은 파일·가짜 Task shape·가짜 외부 모듈만 사용하며 NumPy/Torch를 실제 import하지 않는다. 테스트 명령은 Python `-` stdin에서 정확한 test 파일을 import하고 `unittest`를 실행하는 방식이다. 현재 N4 실행 중이므로 원 캠페인 전체 pytest나 모델 통합시험을 실행하지 않았다.

최종 준비 점검에서 source39+immutable9와 기존 helper·등록·원 V reference를 합친72개 고유 opaque SHA를 실행 전후 대조했다. 세 spec을 원 PLAN과 직접 비교했고 원/새 namespace의 중복·partial·claim 거부 검사를 통과했다. snapshot config/weight 두 경로의 resolved blob·asset root containment·size는 실제 일치했다. 이번 준비 점검에서190MB weight SHA를 새로 계산하거나 모델을 해석하지는 않았으며, 그 기존 pinned SHA는 실제 admission 전후 검사에 남아 있다. tiny runner 최종16 PASS/0.082초, imports NumPy=False/Torch=False였고 default CLI도 reads0/writes0/actual0였다. 별도 실제 실험 로그나 수치 산출물은 만들지 않았다.

검사는 default0IO·seal/admission·원 single-K recipe/다른 ID 거부·partial/claim 보존·동적 F counters·원 V/T 경계·PCA/identity·선택 bundle/route close·metadata-only cost binding·quiet freshness·실제 proof 바인딩·원 stage zero-refit·SHA 변조·write-exclusive·deadline partial 보존을 포함한다.

실제 safe-prefix 숫자 적재, PCA 값, actual model construction/학습 peak/time, fullV FP64, CPU6 parity, B1/B8 성능·메모리, ROOT guard 연동은 아직 검증0이다. 합성 PASS를 실제 통합 PASS나 방법의 성능 이점으로 바꾸어 쓰지 않는다. 독립 read review와 ROOT 전체 source/핀 검토 뒤에도 별도 실제 입장 결정이 필요하다. 한 seed·반복된 개발 V·제한된 학습·K11 초기화 혼재·독립 일반화/수렴 미확인을 유지한다.
