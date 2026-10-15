# matched-native 전체 비용 v2: 물리 입력 manifest 연결 수정

[확인] 최초 비용 실행 `runs/matched_shrunk_native_cost_v1/RESULT.json`은 `PARTIAL_COMPLETED`였고 양 데이터에서 `KeyError: 'arrays'`가 발생했다. 완료 비용 cell 0, timer sample 0, precheck 0이다. 초기 모델·head load는 있었으나 새 학습은 없으며 F의 전후 서명은 같다. 실제 타이밍이나 방법 성능의 실패로 해석하지 않는다.

- 실패 결과 SHA256: `6940e169e2acaa3c27b08c1c80e1e8ae2ddcef944a100af331629165fff09ea9`.
- [원인 검산 기록](audits/MATCHED_NATIVE_COST_INPUT_SCHEMA_FAILURE_IDENTITY_v1.json) SHA256: `408933948ecfa4e171a493f868c136a96b906417e0096c661e95394655d63171`.
- 실행된 v1 runner SHA256: `13a91ce10db4fe9a71f1d67b238bc81aad8d139def9c4309b03a66f4cb0ea255`.
- 동일하게 유지하는 timer helper SHA256: `6779646de2a809c625a2b503b60ac03a64ae72826b15b8efad1d6d2213ae6cc2`.

[확인] 엄격한 `fixed_decoder_span_v1.checked_payload`는 NPZ 파일 SHA뿐 아니라 모든 member의 이름·shape·dtype·배열 SHA가 있는 `record['arrays']`를 요구한다. 원 FIT-target producer의 `physical_bridge`는 파일 path/SHA를 가리키는 alias이므로 `arrays`가 없다. 원 first-Robin/first-Beijing 결과의 `bridge_provenance.input_bridge`가 가리키는 이미 완료된 입력 bridge JSON에는 같은 실제 파일의 전체 manifest가 보존돼 있다. 새 v2 runner가 이를 연결한다.

수정은 새 `run_matched_shrunk_native_cost_v2.py`에서 수행한다. 원 first parent JSON SHA와 완료·TEST 미사용 상태를 확인하고, 그 canonical bridge JSON의 SHA·완료·TEST 미사용 상태를 확인한다. 해당 데이터의 원 DEV split에 대해 artifact가 정확히 한 개여야 한다. 그 artifact의 파일 path와 SHA는 원 physical alias와 같아야 한다. canonical `arrays`의 history·target·observed·current C·origins·FIT mean/std·history observed metadata는 이미 받아들인 normalized raw input 기록과 같아야 한다. 전체 member manifest를 유지한 record를 원 엄격한 loader에 넘긴다. 새 배열 metadata를 추정하거나 `arrays`를 생략하도록 loader를 완화하지 않는다.

파일 SHA, 모든 member 이름·shape·dtype·array SHA 검사와 actual physical/raw mask·origin·normalization parity는 원 그대로다. 같은 원점의 masked physical 값에서 NaN을 유효하지 않은 관측으로 처리하는 계약도 유지한다. 기존 데이터·선택된 E/rho/head/CP, 7경로·28cell·7완전순환·196sample, 원 tolerance, 매 callback 과거 D 적합, 실제 F/native 연산과 physical 반환의 타이머 경계는 바뀌지 않는다. 이 수정으로 학습을 재실행하지 않는다.

원 v1 코드·helper·계획·실패 기록은 보존한다. v2는 새 출력 `runs/matched_shrunk_native_cost_v2`와 새로운 source snapshot을 사용하고 이 addendum 및 실패 원인 기록을 bind한다. 원 [비용 계획](PLAN_MATCHED_SHRUNK_NATIVE_COST_KO_v1.md)을 수정하지 않는다. source-only 단계에서 synthetic strict manifest/변조/누락/마스크 fixture 및 실제 JSON metadata-only canonical 연결을 검산하며, 실제 NPZ·모델을 읽거나 실제 F/학습/비용을 실행하지 않는다. root의 source 검토 후 v2 actual을 수행한다. 이 문서 작성 시 v2 actual 비용은 아직 미실행이다.
