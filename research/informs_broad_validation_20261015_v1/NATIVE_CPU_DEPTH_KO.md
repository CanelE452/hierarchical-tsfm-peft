# NO_F native CPU fresh depth 보강

이 문서는 기존 GPU 실험의 재개·완료 판정이 아니라, 별도 device 조건의 새 CPU 실험 9개를 준비한다. 실제 실행 여부는 새 실행 영수증으로 판단한다. 계획과 합성 테스트 통과는 실제 학습·예측 또는 수렴의 증거가 아니다.

## 대상과 비교 계약

- Jena·Electricity·PeMS08 각각 DIRECT_G·DIRECT_B2·DIRECT_B4. Beijing의 완료된 경로는 중복하지 않는다.
- ID는 `primary_{dataset}_E1_{family}_k0_s92601__cpu_fresh_e3`이다. ID/registry의 k0는 비압축 native 구분이며 실제 spec의 latent_k는 원 채널 수 C이다.
- 초기 SOURCE_CONTRACT.spec에서 run_id, device=cpu, epochs=[0.25,1,3], stage_label, provenance만 바꾼다. seed92601, lr0.001, microbatch4, effective_batch16, primary/base_parent, A50 search·Astop checkpoint 선택·선택 epoch로 새 A60 refit은 기존 API대로 유지한다.
- 원 GPU weights·optimizer·RNG·search/refit state를 읽거나 재사용하지 않는다. 모델 정체성은 `NO_FOUNDATION`, foundation 인자는 None이며 학습/실제 V 예측의 F stream counter를 0으로 확인한다.
- CPU 조건을 기존 GPU `__e1`/`__e3` 완료나 수렴 증거로 합치지 않는다. 기존 sealed E2 계획·부모 선택은 바꾸지 않는다.

## 실행 경계

기본 CLI는 stdlib metadata만 읽으며 NumPy·Torch·원자료·모델을 import/load하지 않는다. 원 출처 JSON과 registry, DATA_CONTRACT, 고정 source33 SHA만 검증한다. ROOT가 저장된 plan의 proposed_registry_rows를 기존 registry_extension.register API로 별도 등록한 뒤, ROOT의 명시적 trigger에 따라 fresh CPU subprocess에서 `--execute --plan-file <saved plan>`을 실행한다. worker는 registry를 쓰지 않는다.

실제 실행은 기존 data_contract.load_task의 합법 safe-prefix와 fit_base.fit_base, prediction_cache.predict_parent의 V_select만 사용한다. 이 실행 단계에서는 학습용 자료 및 자체 생성한 새 native artifact를 기존 API가 읽는다. 기존 prediction API의 trusted campaign artifact 로딩은 weights_only=False/map_location=cpu이며, 이를 weights_only=True 검증으로 표현하지 않는다. 외부/출처 불명 pickle, foundation 생성·추론, GPU lease, TEST/T, 설치·다운로드·commit/push는 없다.

처음 상한은 wall1800초, CPU threads2, RSS4GiB, C+R30GiB 및 pending512MiB 여유, 기존 campaign 48시간 deadline 이내이다. 각 route 전에 source/selection/정확한 사전등록 scientific identity/빈 출력·claim 경로/저장 공간을 재검사한다. claim은 exclusive 생성하며 기존 출력이나 부분 상태를 덮거나 재개하지 않는다. route 실패는 private traceback에 보존하고 독립 route를 계속한다. 공개 실패 reason은 public/cache/asset 절대경로를 비공개 처리한다.

RSS 감시가 자체 worker 한도를 넘으면 RESOURCE_STOP에 PID·현재 run ID·예상 exit98을 기록하고 os._exit(98)로 자체 프로세스를 끝낸다. 이 경우 STARTED/RESOURCE_STOP과 부분 산출물만 남고 최종 RESULT가 없을 수 있다. ROOT는 STARTED의 PID와 실제 프로세스 exitcode/종료 여부를 별도 확인해 미완료로 기록해야 한다. work-boundary deadline은 fit/inference 기존 API로 전달하며 부분 결과를 완료라 부르지 않는다. 다음 정밀 GPU 비용 측정 전에 CPU worker 종료를 확인한다.

registry는 GPU 병행 작업에 의해 외부 mutable status나 새 row가 바뀔 수 있다. worker는 전후 bytes SHA와 외부 변경 ID를 기록하며 공통 row의 scientific identity가 바뀌면 실패로 보존한다. registry 수정은 ROOT의 단일 writer 경계에서만 한다.

## 검증과 시간 근거

`python -X utf8 -B -m unittest test_native_cpu_depth -v`만 명시적으로 실행한다. 14개 합성 fixture는 NO_F·F0, 미등록, 잘못된 identity, selection lock, 기존 경로, source 변경, spec 변경, 중복 route, 재실행, wall 상한, 경로 privacy, 한 route 실패 후 독립 계속을 검사한다. 기본 root pytest나 실제 data/model 기반 smoke는 실행하지 않는다.

기존 DIRECT_G CPU 3epoch 약 8초의 근거만 있으며 DIRECT_B2/B4 CPU 시간은 미측정이다. 따라서 9개 전체 완료 시간은 불명확하다. 첫 실제 실행에서 epoch/선택·updates·exposure·자원·실패를 남기고, 30분 안에 끝나지 않은 경로는 그대로 미완료로 보고한다. 기존 학습 API가 생성한 search/refit RECEIPT의 exposure가 원 증거이며 wrapper가 새로운 telemetry를 꾸며내지 않는다.
