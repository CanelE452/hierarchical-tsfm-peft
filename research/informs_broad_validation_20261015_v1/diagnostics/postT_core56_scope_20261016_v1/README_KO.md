# 원 core56과 별도 post-T CPU 결과의 대응 범위

**[확인] 원 GPU 장부의 8/56은 그대로다. 장치 변경을 명시한 과학 조건 대응은 기존 GPU8·CPU fresh9의 서로 다른 원 조건17개에, 검산까지 끝난 Jena B0/B2 K6 두 조건을 더해 19/56이다. 이 snapshot의 잔여는37개다.** 기존 등록 상태를 바꾸거나 CPU 결과를 원 GPU 실행으로 소급하지 않았다.

[56행 대응표](CORE56_CORRESPONDENCE.csv)와 [입력 SHA·실제 증거·검증 결과](INPUT_BINDINGS.json)에 각 원 ID·family·C/K·완료 actual ID·검색 최대 epoch·선택 refit epoch·미완료·계획 대응을 남겼다. 원 요구는 [MASTER 공개본 7.3](../../plan/MASTER_PLAN_PUBLIC.md#73-학습량과-최적화)의 0.25→1→3epoch 선별이며, 이 장부의 원 최소56 범위는 [동결된 깊이 CSV](../final_minimum_coverage_1792094400000000000/E1_MINIMUM_NEURAL_DEPTH56.csv)를 그대로 재사용했다.

## 숫자의 의미

```text
구분                                         현재 실제 또는 시나리오
원 GPU에서 3epoch 검색·선택 artifact 완료       8 / 56 (원 장부 유지)
기존 별도 CPU fresh native 조건 대응           9 / 56 (GPU8과 원 ID 교집합0)
기존 any-device 대응                           17 / 56 (기존 검증 범위 재사용)
새 Jena B0/B2 K6 대응                          2 / 56 (기존17과 교집합0)
현재 완료된 any-device 조건 대응              19 / 56
현재 남은 원 core 조건                        37 / 56
rolling15 중 미래 신규 core 대응 가능          8 (현재 완료에 합산하지 않음)
rolling15 중 기존 core 중복                    1
rolling15 중 core56 밖                         6
모든 미래 신규8이 성공하는 경우에만            27 / 56, 잔여29 (조건부 시나리오)
```

기존17은 원 깊이 장부와 기존 artifact 검증 진술을 재사용하고, 해당 BASE의 작은 메타데이터·source/artifact 선언 SHA만 연결했다. 모델을 다시 읽거나 checkpoint·배열·점수·시간을 재감사한 결과가 아니다. 원56 CSV 및 원 registry471을 고치지 않았다.

## 새로 추가한 실제 대응2

- `primary_jena_E1_B0_k6_s92601` ← `postT_cpu_jena_E3_B0_k6_s92601_v1`: [BASE](../../postT_V_only_cpu_lag_20261016_v1/runs/postT_cpu_jena_E3_B0_k6_s92601_v1/BASE_FIT_RECEIPT.json), [전체 V43 FP64](../../postT_V_only_cpu_lag_20261016_v1/diagnostics/root_cpu_sequential_v1_j01_lag_rescore/RESULT.json), [새 CPU6 replay](../../postT_V_only_cpu_lag_20261016_v1/diagnostics/root_cpu_sequential_v1_j02_lag_replay/RESULT.json). 검색 최대3epoch/1863updates/29769exposures; A_stop 선택 후 fresh A60 refit은 3.0epoch/2254updates/36060exposures다.
- `primary_jena_E1_B2_k6_s92601` ← `postT_cpu_jena_E3_B2_k6_s92601_v1`: [BASE](../../postT_V_only_cpu_lag_20261016_v1/runs/postT_cpu_jena_E3_B2_k6_s92601_v1/BASE_FIT_RECEIPT.json), [전체 V43 FP64](../../postT_V_only_cpu_lag_20261016_v1/diagnostics/root_cpu_sequential_v1_j03_lag_rescore/RESULT.json), [새 CPU6 replay](../../postT_V_only_cpu_lag_20261016_v1/diagnostics/root_cpu_sequential_v1_j04_lag_replay/RESULT.json). 검색 최대3epoch/1863updates/29769exposures; A_stop 선택 후 fresh A60 refit은 1.0epoch/752updates/12020exposures다.

두 조건은 dataset/fold/L512/H48/C21/K6/seed92601·A50/A_stop/A60 경계·원 source6·frozen foundation SHA·검색/refit 정규화·초기화 SHA·V의 target/mask/origins/columns 지문을 메타데이터로 대조했다. 원 loss를 구현한 source와 lr0.001/effective16/micro4/row128/checkpointFalse recipe가 같고, 바뀐 run ID·deviceCPU·검색 일정·stage/provenance는 별도 기록했다. CPU와 CUDA의 수치 연산·실행 비용이 같거나 동일한 장치 실험이라는 주장은 아니다.

Jena B0는 검색3epoch의 마지막이 best이고 BASE의 `UNDERTRAINED_OR_BUDGET_LIMIT`을 그대로 유지한다. B2는 검색3epoch를 끝냈지만 A_stop이 선택한 fresh refit은1epoch다. 검색 최대 epoch와 선택 refit epoch, 충분 수렴은 서로 다른 판단이다. 전체 V43 FP64와 CPU6 replay는 저장 수치·artifact 재현 검증이며 독립 일반화 검증이 아니다. 두 조건의 B1/B8 비용 산술 PASS도 [별도 감사](../../postT_V_only_cpu_lag_20261016_v1/diagnostics/root_cpu_cost_arithmetic_20261016_v1/AUDIT_RESULT.json)로만 연결하고, 이 장부의 학습 깊이 판단을 대신하지 않는다.

## 진행 계획15의 중복 없는 대응

현재 완료 수에 넣지 않은 미래8은 아래와 같다. [E4 계획](../../postT_V_only_cpu_multik_20261016_v1/PLAN.json)·[Beijing required 계획](../../postT_V_only_cpu_required_minima_20261016_v1/PLAN.json)·[family4 계획](../../postT_V_only_cpu_family_minima_20261016_v1/PLAN.json)의 `upstream_registered_id`를 원56 집합과 exact join했다. 실제 진행 중 결과 디렉터리와 mutable 상태는 읽지 않았다.

```text
primary_beijing_E1_B0_k6_s92601 <- postT_cpu_beijing_E4_INDEPENDENT_B0_k6_s92601_v1
primary_beijing_E1_B0_k11_s92601 <- postT_cpu_beijing_E4_INDEPENDENT_B0_k11_s92601_v1
primary_beijing_E1_B2_k6_s92601 <- postT_cpu_beijing_E1_B2_k6_s92601_required_v1
primary_beijing_E1_B2_k11_s92601 <- postT_cpu_beijing_E1_B2_k11_s92601_required_v1
primary_jena_E1_B4_k6_s92601 <- postT_cpu_jena_E1_B4_k6_s92601_family_v1
primary_jena_E1_ADAPTS_BOLT_PORT_k6_s92601 <- postT_cpu_jena_E1_ADAPTS_BOLT_PORT_k6_s92601_family_v1
primary_jena_E1_LORA_R8_k0_s92601 <- postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1
primary_beijing_E1_B4_k11_s92601 <- postT_cpu_beijing_E1_B4_k11_s92601_family_v1
```

E4 독립 B0 K3는 기존 Beijing B0 K3와 중복이므로 신규 완료 수를 늘리지 않는다. E4 독립 B0 K6/11은 원 E1 B0 K6/11에 대응하지만 shared/max parent2 및 prefix/nested 출력은 원56에 없다. Beijing lag4와 이미 끝난 Jena lag2도 E3이며 원 core56을 채운 것으로 합산하지 않는다. LoRA는 원 등록 이름 `LORA_R8/K0`에 연결하고 실제 학습 spec의 `LORA/K21`은 C21개 F stream의 구현 표현으로 따로 기록했다. 미래8을 인정하려면 각각 full search3·선택 fresh A60 refit·full V·독립 FP64/replay까지 실제로 끝나야 한다. 계획·probe·search 중간 완료만으로 이 표를 갱신하지 않는다.

## 현재 남은37: exact 원 ID

```text
primary_jena_E1_LORA_R8_k0_s92601
primary_electricity_E1_LORA_R8_k0_s92601
primary_pems08_E1_LORA_R8_k0_s92601
primary_jena_E1_ADAPTS_BOLT_PORT_k6_s92601
primary_electricity_E1_ADAPTS_BOLT_PORT_k81_s92601
primary_pems08_E1_ADAPTS_BOLT_PORT_k43_s92601
primary_beijing_E1_B0_k2_s92601
primary_beijing_E1_B0_k6_s92601
primary_beijing_E1_B0_k11_s92601
primary_beijing_E1_B2_k2_s92601
primary_beijing_E1_B2_k6_s92601
primary_beijing_E1_B2_k11_s92601
primary_jena_E1_B0_k5_s92601
primary_jena_E1_B0_k11_s92601
primary_jena_E1_B0_k21_s92601
primary_jena_E1_B2_k5_s92601
primary_jena_E1_B2_k11_s92601
primary_jena_E1_B2_k21_s92601
primary_electricity_E1_B0_k81_s92601
primary_electricity_E1_B0_k161_s92601
primary_electricity_E1_B0_k321_s92601
primary_electricity_E1_B2_k81_s92601
primary_electricity_E1_B2_k161_s92601
primary_electricity_E1_B2_k321_s92601
primary_pems08_E1_B0_k43_s92601
primary_pems08_E1_B0_k85_s92601
primary_pems08_E1_B0_k170_s92601
primary_pems08_E1_B2_k43_s92601
primary_pems08_E1_B2_k85_s92601
primary_pems08_E1_B2_k170_s92601
primary_beijing_E1_B4_k11_s92601
primary_jena_E1_B4_k6_s92601
primary_jena_E1_B4_k21_s92601
primary_electricity_E1_B4_k81_s92601
primary_electricity_E1_B4_k321_s92601
primary_pems08_E1_B4_k43_s92601
primary_pems08_E1_B4_k170_s92601
```

## 미래8이 모두 성공하는 경우에만 남는29

아래는 현재 상태가 아니라 조건부 집합 차이다. small9는 Beijing B0/B2 K2 두 조건과 Jena B0/B2 K5/11/21 여섯 조건·B4 K21 하나이며, Electricity10·PeMS08 10이 별도로 남는다.

```text
primary_electricity_E1_LORA_R8_k0_s92601
primary_pems08_E1_LORA_R8_k0_s92601
primary_electricity_E1_ADAPTS_BOLT_PORT_k81_s92601
primary_pems08_E1_ADAPTS_BOLT_PORT_k43_s92601
primary_beijing_E1_B0_k2_s92601
primary_beijing_E1_B2_k2_s92601
primary_jena_E1_B0_k5_s92601
primary_jena_E1_B0_k11_s92601
primary_jena_E1_B0_k21_s92601
primary_jena_E1_B2_k5_s92601
primary_jena_E1_B2_k11_s92601
primary_jena_E1_B2_k21_s92601
primary_electricity_E1_B0_k81_s92601
primary_electricity_E1_B0_k161_s92601
primary_electricity_E1_B0_k321_s92601
primary_electricity_E1_B2_k81_s92601
primary_electricity_E1_B2_k161_s92601
primary_electricity_E1_B2_k321_s92601
primary_pems08_E1_B0_k43_s92601
primary_pems08_E1_B0_k85_s92601
primary_pems08_E1_B0_k170_s92601
primary_pems08_E1_B2_k43_s92601
primary_pems08_E1_B2_k85_s92601
primary_pems08_E1_B2_k170_s92601
primary_jena_E1_B4_k21_s92601
primary_electricity_E1_B4_k81_s92601
primary_electricity_E1_B4_k321_s92601
primary_pems08_E1_B4_k43_s92601
primary_pems08_E1_B4_k170_s92601
```

## 원 계획·안전 범위

이 대응표는 원 등록 nonCOMPLETE192개 전체를 ‘필수 미달192’라고 부르지 않는다. core56은 지정된 main4 자료·dataset/K·선별 seed의 별도 부분집합이다. 3epoch 선별 대응19도 원 GPU 부족48, M5 자료 부족, 더 큰 학습량·추가 seed·다른 K·E2/E3/E4·비용 전체·게시 목표를 완료로 만들지 않는다. 원 MASTER 전체는 PARTIAL이다. 원 [RESUME](../../RESUME.md)의 명령과 CSV의 `original_resume_reference_PRE_SELECTION_ONLY`는 pre-selection recipe 참조이며, 현재 잠긴 T를 본 뒤 곧바로 재선택·재학습하라는 명령이 아니다. 새 development/별도 assessment 설계와 자원 감사가 필요하다.

이번 snapshot은 post-T V-only 개발 대응으로, 원 source39/registry/selection/T 결과를 보존했다. 새 T target·선택·Task/배열/모델/가중치·통계 gzip·학습·추론·비용 계측·Git 동작은0이다. 기존 safeFIT 물리 역직렬화 이력까지 세션 전체 T-byte 접근0으로 바꾸어 말하지 않는다. 읽은 입력은 공개 작은 JSON/CSV·문서·source SHA이며 전후 모두 같은 byte SHA임을 확인했다.
