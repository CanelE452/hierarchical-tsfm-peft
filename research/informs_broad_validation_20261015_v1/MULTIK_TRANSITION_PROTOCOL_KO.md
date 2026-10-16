# E4 K 전환·저장 비용 보완

MASTER_PLAN_PUBLIC §11(공유 K), §12, §14의 기존 요구를 측정한다. 새 방법·재학습·T 평가가 아니다. 실제 GPU 실행은 ROOT가 기존 owner lease를 차용하는 child로 수행하며, 이 구현 담당자는 CPU 합성 시험만 실행했다. 기존 379 CPU suite는 이 새 모듈/시험보다 먼저 수행되었으므로 이 코드까지 검증했다고 확대하지 않는다.

## 고정 조건과 경로

Beijing/Electricity 각각 K={ceil(C/4),ceil(C/2),C}, B=1/8, 동일 V_select의 앞 B개 서로 다른 origin의 과거 physical/mask 창만 사용한다. target_window를 호출하지 않는다. `--make-plan`은 기존 initial checkpoint ID만 명시하는 계획이며, 각 ID를 다른 완료 parent로 바꾸려면 T 선택 전에 정확한 새 alias/receipt를 별도로 고정해야 한다. 이 모듈은 registry나 원 결과를 수정하지 않는다.

- `shared`: 실제 공동 K 학습 checkpoint 1개를 Route 1개에 로드한다. 저장 spec의 k_grid가 정확히 세 K인지 먼저 검사한다.
- `posthoc`: 최대 K 전용 학습 checkpoint 1개를 Route 1개에 로드한다. 저장 spec.k_grid=[C]를 확인한 뒤 runtime `ForecastAdapter.k_grid`만 세 K로 허용한다. 이는 원 `ordered_pilot.predict_prefix`의 배포용 `spec[k_grid]=sorted(set(trained_grid+[K]))`와 같은 허용-prefix 확장이다. 저장 spec·E/P/D tensor를 수정하지 않는다.
- `nested`: 전체 PCA basis artifact 1개를 Route 1개에 로드한다.
- `independent_hot`: 독립 B0 checkpoint 세 개·Route 세 개가 사전 상주하며 frozen F pool은 하나다.
- `independent_cold`: 다른 독립 B0로 바꿀 때 이전 Route를 deactivate/close/release한 다음 새 artifact로 Route를 생성한다. frozen F는 계속 상주한다. OS 파일/페이지 캐시는 통제하지 않으므로 disk-cold latency라고 부르지 않는다.

single-Route의 실제 전환 속성은 `Route.latent_k`다. frozen `deployment.Route._predict_normalized`가 encode(x,K), F의 실제 K rows, decode(...,K)를 호출한다. nested는 E[:,:K]를 사용한다. 별도 K 모델 세 개를 숨겨 상주시키지 않는다. 허용 K 속성 변경과 tensor 변경을 분리해 shared/posthoc/nested 및 hot model-state hash 전후 불변을 검사한다. cold는 각 로드의 exact artifact 계약과 canonical parity를 사용한다.

## 실제 추론 검산 후 측정

각 전략에서 K마다 원 `route_catalog.build_route`의 정확한 canonical Route를 생성하고 동일 physical 창을 실제 예측한다. runtime 전환 결과가 canonical 결과와 atol=1e-5/rtol=1e-4 이내이며 양쪽 actual logical/physical F stream이 B*K인지 검사한다. 실제 Fmax 후 자르기를 작은 K 비용으로 세지 않는다. 저장 예측 배열을 읽는 시간을 추론 비용으로 사용하지 않는다. canonical accuracy manifest/receipt/artifact/foundation/data/scaler 계약도 묶는다.

parity가 실패한 전략은 그 전략만 실패이며 다른 전략은 계속할 수 있다. canonical/parity 이전 또는 중간에 자원 상한에 도달하면 각 transition의 실측 완료/미완료 수를 그대로 남긴다. 예측이 끝나지 않았거나 parity가 실패한 trial은 완료 수에 넣지 않는다.

canonical Route 생성자의 사전 검사만으로 예측을 허용하지 않는다. 생성자가 반환한 직후, 첫 physical 예측을 시작하기 전에 owner를 다시 검사한다. 생성 중 상한을 넘겼으면 해당 Route를 finally에서 닫고 canonical 예측 0회로 중단한다. 이 경계는 비퇴화 CPU fixture에서 실제 constructor 반환 후 deadline을 넘기는 주입으로 RED→GREEN 검증한다. 기존 V2 증거는 보존하며 이 보완을 실제 GPU 검증으로 확대하지 않는다.

## 타이머와 반복

3 K 사이 6개 directed transition 각각 warmup 10회, timed 20회를 실행한다. 매 round에서 transition 순서를 회전시켜 6 rounds마다 각 transition이 모든 위치를 한 번씩 차지한다. 각 trial 전에 from-K를 맞추는 준비 시간은 outside-timed 필드로 따로 기록한다.

주 관측값 `transition_plus_physical_seconds`는 실제 K 전환의 deactivate/configuration/release/load/activate와 CPU physical 입력→창 내 fill/A60 정규화→P/E→실제 F(K)→D→원단위 역변환→CPU 반환·동기화를 포함한다. `physical_pipeline_seconds`는 frozen Route의 전체 물리 파이프라인 타이머다. K 설정·로드·활성/비활성 구간은 각각 동기화 탐침이 있는 보조 타이머로 기록하며, 보조 합계가 whole과 같다고 주장하지 않는다. 기존 PeMS warmed steady-state predict 비용과 합쳐 평균내지 않는다.

cold load에는 현재 Route constructor의 artifact hash/read, normalization/PCA regeneration·validation, parameter state load, device placement가 포함된다. 이는 최적화된 배포 로더나 순수 디스크 I/O 측정이 아니다. 초기 전체 모델/F 로드는 별도 필드이며 매 transition 비용에 다시 넣지 않는다. 모든 trial은 자체 파이프라인을 실제 호출한다.

`actual_trials/trial_*.json`은 수치 예측·canonical 검사가 완료된 trial마다 새 파일로 보존한다. 각 단계에서 owner를 확인한다: trial 시작 전, from-K 준비 반환 후, target-K 전환 반환 후, 물리 예측·동기화 반환 후, measurement 반환 후, 각 후처리 검산 후, 최종 close·동기화 후, 완료 판정 직전. 긴 함수를 강제 종료하지 않고 반환 뒤 한계를 검사한다. 자원 제한이 확인되면 새 예측·다음 strategy를 시작하지 않는다.

수치 실행 상태·자원 상태·후처리 검증 상태는 별도 필드다. 마지막 180번째 예측이 deadline 뒤에 반환되어도 실제 수행한 180개 수치/예측 hash를 지우지 않는다. `numeric_execution_status=COMPLETE_REQUESTED_TRIALS`이면서 `resource_status=DEADLINE_OVERRUN`, 최종 `exact_minimum_protocol_complete=false`일 수 있다. ledger의 `actual_complete/remaining`은 실제 수치 완료/미실행, `completed_within_resource/remaining_within_resource`는 자원 내 완료/미충족이다. 따라서 마지막 예측 초과의 numeric remaining=0과 resource-valid remaining=1을 구분한다. close에서 처음 초과가 발견돼도 final protocol은 미완료다.

NUMERIC_EXECUTION.json은 memory·tensor-state·summary 검산과 close 이전에 저장하는 잠정 수치 영수증이다. 최종 완료 선언을 하지 않는다. `_memory`, model-state 검산, summary가 실패해도 이미 저장된 rows와 원 ledger를 덮지 않고 해당 후처리 오류를 추가한다. 실제 trial JSON·TRIALS.csv·최종 rows의 일치를 따로 확인한다. 최종 판정은 RESULT.json/BATCH_RECEIPT.json을 사용한다. warmup/timed 수를 줄여 실행해도 최소 protocol 완료로 세지 않는다.

전환 후 owner 검사는 예측 전에 필요하므로 instrumented whole에 그 controller 검사 시간이 들어간다. cold factory 안의 owner 검사 시간도 누적 계측한다. 서로 겹치지 않는 `switch_execution_segment_seconds`(switch 반환까지의 wall에서 그 안의 factory guard만 제외), `physical_prediction_segment_seconds`, `controller_guard_seconds_inside_whole`(factory 내부+두 구간 사이), `other_clock_overhead_seconds`의 합이 관측된 guarded whole과 같다. 시험으로 이 분해를 확인한다. 원 switch 보조 타이머·`physical_pipeline_seconds`는 각각 위 구간 안에 포함되는 별도 탐침이므로 위 합에 다시 더하지 않는다.

guard를 뺀 보조값도 제공하지만 다른 동기화 탐침·constructor validation까지 제거한 순수 배포 시간이나 공정 우위로 주장하지 않는다. `physical_pipeline_seconds`는 frozen Route 자체의 CPU physical 입력부터 CPU 반환까지이며 두 구간 사이 owner 검사 바깥이다. 기존 warmed predict와 guarded transition scope를 혼합하지 않는다.

CPU 합성 fixture는 persistence 대신 비영(非零) 미래 drift와 채널별 다른 과거 패턴, 명시적인 E/D를 사용한다. K별 출력이 실제로 다름을 먼저 확인한 뒤, E의 잘못된 column 선택 및 금지된 D.reshape(7*Kmax,C)[:7*K] 절단이 B*K stream 수를 유지해도 canonical mismatch로 잡히는지 검사한다. 실제 데이터/백본/보호 T를 사용한 검증으로 확대하지 않는다.

## 저장량·상주량

SHA 검산된 실제 artifact 경로를 중복 제거해 파일 개수·실제 byte를 집계한다. 같은 SHA를 가진 별도 파일 두 개는 실제 저장량에서 두 개로 세고 content-unique 개수는 별도로 제공한다. 공통 frozen F 저장량은 adapter 비교에서 제외했다고 명시한다. hot의 세 실제 모델과 shared의 한 모델을 구분하며 parameter/buffer 저장소 포인터로 상주 weights를 중복 제거한다. CUDA allocated/reserved peak와 RSS는 별도다. training peak를 포함하지 않는다.

정확한 fresh-process 비교는 전략 하나당 새 guarded worker를 호출한다. `--strategy all`은 같은 프로세스 내 순차 전략 측정이므로 fresh-process memory라고 부르지 않으며 개별 실패/자원 중단이 다른 전략의 완료를 대신하지 않는다.

## CLI

```powershell
python -X utf8 -u measure_multik_transition.py --make-plan --dataset beijing --channels 11 --output <새_plan.json>
python -X utf8 -u measure_multik_transition.py --make-plan --dataset electricity --channels 321 --output <새_plan.json>
```

ROOT가 승인된 기존 GPU owner 환경을 child에 전달한 다음 각 전략/B에 새 출력 폴더를 지정한다. 이 모듈은 새 GPU owner를 만들지 않는다.

```powershell
python -X utf8 -u measure_multik_transition.py --worker --plan-json <plan.json> --public-root <campaign> --cache-root <private_cache> --model <고정_small_snapshot> --device cuda --batch 1 --strategy shared --warmups 10 --timed-trials 20 --output <새_output_directory>
```

B8 및 다른 전략도 같은 고정 plan을 사용한다. 실제 모델이 없거나 canonical K 전환을 지원하지 않으면 자료/구현 부재로 해당 전략을 미완료 처리하며 가능한 척하지 않는다. 성능이 음성이라는 이유로 전환 측정을 생략하는 gate는 없다.
