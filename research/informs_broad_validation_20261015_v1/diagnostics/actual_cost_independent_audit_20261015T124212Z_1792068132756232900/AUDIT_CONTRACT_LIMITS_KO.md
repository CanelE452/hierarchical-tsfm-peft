# 저장 비용 감사의 계약과 한계

이 문서는 최초 B1 감사와 새 최종 감사에 공통으로 적용한다. 원 타이머·모델·데이터·배열을 재실행하거나 원 파일을 수정하지 않았다.

1. 기록된 whole_seconds는 route activation/synchronization이 끝난 뒤의 물리 예측 경로다. balanced route.activate의 LoRA merged-weight CPU→GPU 복사와 route.deactivate의 unmerge/backbone hash는 타이머 밖이다. 전환 비용까지 포함한 전체 배포 latency를 측정했다고 확대하지 않는다. 근거: measure_costs.py의 measure_balanced, deployment.py의 activate/deactivate/predict_physical.

2. resident_weights_cpu_MiB는 등록 module parameter/buffer 바이트다. LoRA의 비등록 CPU _originals와 _merged_weights list를 포함하지 않으므로 전체 CPU 배포 state가 아니다. allocator peak와 공존 모델 memory는 fresh/balanced scope로 분리하며 RSS_after_close는 추론 peak로 해석하지 않는다.

3. Raw row parity와 LoRA merged/unmerged parity, exact backbone restore는 public 기록의 PASS/허용오차/SHA 및 CSV prediction SHA와의 일치만 검산했다. 배열 allclose와 tensor 복원을 독립 실행하지 않았다. 단일 max_abs만으로 relative allclose를 재판정하지 않았다.

4. 최초 AUDIT_B1의 ridge expected_prediction_model_SHA 한 건은 감사자의 서로 다른 SHA 층 직접동일성 가정이 틀린 경우였다. route_catalog.py:54는 native fit model SHA, prediction_cache.py:164는 artifact/model_state/normalization composite를 V model SHA로 기록한다. 새 B1_V2는 nativeSHA↔public fit receipt, compositeSHA↔V manifest binding 및 공통 artifact/scaler SHA로 분리해 통과했다. 최초 FAILED 및 원 파일을 보존했다. weights/private spec를 열어 독립 재생한 증거는 아니다.

5. fresh와 balanced의 같은 route/row/input/mask 비율은 별도 범위 차이 진단이다. Raw-small B1은 약2.1893배, LoRA는2.0084배이며 원인은 미확정이다. 공유 session·전환·시스템 상태를 후보로 남기고 오류로 단정하지 않는다. 주요 방법 비교는 같은 scope 및 명시된 row variant 안에서 수행한다.

6. 저장 trial의 median/mean/min/max/p90와 stage 열 재집계·F 호출 산술·순환순서·공개 SHA 계보 검산이다. 타이머 정확도/장치상태/allocator/weights 실제 bytes·독립 최종 T·연구 일반화 검증은 아니다.
