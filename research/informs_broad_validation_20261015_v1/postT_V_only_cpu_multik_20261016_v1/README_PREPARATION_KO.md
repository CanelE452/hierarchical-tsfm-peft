# 다중 K CPU 보강 wrapper 준비 기록

**준비 단계이며 실제 모델·배열·학습·추론·자식 프로세스 실행은 0이다.** 이 문서는 실행 결과나 전체 연구 완료 보고서가 아니다. 최종 `PLAN.json`과 별도 `REGISTRY.csv`는 ROOT가 코드 검토 후 봉인한다. 현재 진행 중인 Jena 학습, 비용 및 고정 PCA 큐를 이 준비 작업이 중단하거나 재시작하지 않는다.

원 E4 방법을 과거에 관찰한 T와 분리된 CPU 개발 조건에서 검토하기 위한 wrapper다. 원 `source39`, registry, family lock, 선택 잠금, score/cost/T 고정 자료를 보존하며 원 GPU 최소 학습 미달을 완료로 바꾸지 않는다. 이미 본 V를 다시 사용하는 개발 결과이므로 새로운 독립 일반화 검증이 아니다. 연구 성공과 실행 완료는 별개다.

## 고정한 비교

Beijing C11/L512/H48, seed92601, 기존 family lock의 B0와 K `[3,6,11]`을 유지한다. 이전 V/T 점수로 family를 다시 고르지 않는다. 새 적합은 다음 5개다.

```text
postT_cpu_beijing_E4_ORDERED_MULTI_K_k11_s92601_v1
postT_cpu_beijing_E4_MAXK_PCA_TRAINED_k11_s92601_v1
postT_cpu_beijing_E4_INDEPENDENT_B0_k3_s92601_v1
postT_cpu_beijing_E4_INDEPENDENT_B0_k6_s92601_v1
postT_cpu_beijing_E4_INDEPENDENT_B0_k11_s92601_v1
```

등록된 역할 이름과 실제 모델 family를 구분한다. 원 `make_specs`와 적합 bundle의 실제 family는 B0다. 공유 모델은 `k_grid=[3,6,11]`, 최대 K 모델은 **`k_grid=[11]`을 반드시 유지**한다. 원 `create_model`은 이 필드가 있을 때 full PCA로 초기화한다. 독립 K11 B0는 원 E1 recipe대로 `k_grid`가 없어 항등 초기화한다. 이 차이를 공유 학습의 효과로 전부 해석하지 않는다.

학습 recipe는 원 A50 검색, A_stop 선택, fresh A60 재적합을 유지한다. 원 `[.25,1,3]` 검색, AdamW lr.001/weight_decay0/clipping1, effective16/micro4, frozen Bolt-small row128/checkpoint=False를 사용한다. 초기 epoch0가 선택돼 실제 refit update가0일 수 있다. 이 경우도 원 값을 보존하며 학습된 checkpoint라고 과장하지 않는다. 마지막 checkpoint가 best이거나 계속 감소하는 경우 원 training_status/last_cp_best를 남기고 수렴을 주장하지 않는다.

V 출력은 **12개**다: 독립3, 공유3, 최대 K 부모의 작은 사후 prefix2와 full-C1, nested PCA3. 원 고수준 bundle은 `posthoc C`와 `max_only C`에 별도 alias를 만들어 독립3까지 포함하면13개가 될 수 있다. 여기서는 같은 full-C 부모/실행 slot을 `max_only C`로 한 번만 평가하여 3+3+3+3=12로 정의한다. 추가 max-C 중복을 새 증거로 세지 않는다.

독립3 V는 원 `prediction_cache.predict_parent(...,'V_select')`를 사용해 새 namespace의 `predictions/<ID>/V_select`에 저장한다. 나머지9는 원 `ordered_pilot.predict_prefix`로 `ordered_predictions/<ID>`에 저장한다. 작은 독립 K 모델은 원 prefix API의 full-C parent 조건을 만족하지 않아 이 경로 분리가 필요하다. `route_catalog.build_route`는 `_E4_` ID 문자열이 아닌 ordered manifest 존재 여부로 경로를 고르므로 E4 독립 ID도 원 standard route로 해석한다. 원 코어를 수정하지 않는다.

## 실행 모드와 검사

`multik_runner.py` 기본 실행은 stdlib metadata만 출력한다. 세 원 root를 명시하면 봉인된 PLAN의 파일 SHA와 작은 메타데이터를 검증한다. 실제 작업은 `--execute --mode {probe,fit,prefix,nested} --run-id ... --plan-sha ... --public-root ... --cache-root ... --asset-root ... --deadline ... --output ...`를 모두 명시해야 한다.

- `probe`: shared/max 중 한 조건만 fresh A50에서3 effective updates/48 samples 후 폐기한다. 전체 legal origins의 loss 정규화와 permutation seed를 유지한다. shared K순환과16개씩의 노출, max K11의48노출, finite gradients, frozen F 상태/parameter gradients 부재를 확인한다. 예상 실제 logical/physical streams는 shared320/max528이고 물리 호출은 각각12회다. probe state를 적합에 재사용하거나 저장하지 않는다.
- `fit`: 한 조건의 원 `fit_base.fit_base`만 호출한다. 완성 receipt는 원 search의0/.25/1/3 endpoint와 실제3epoch 노출, 선택된 refit 노출을 검증한다. V는 별도 모드다. 예산 중단과 incomplete receipt를 완료로 바꾸지 않는다.
- `prefix`: 독립 또는 shared/posthoc/max 한 V 출력만 계산한다. 기존 selected parent의 spec/완성 receipt를 검증하며, 실제 F counter 증가가43×K인지 확인한다.
- `nested`: 원 A60 full-PCA basis의 한 prefix V만 계산한다. 같은 새 namespace 안의 고정 nested basis는 원 API의 source/data/artifact 계약이 같은 경우에만 공용으로 사용한다. 학습 checkpoint의 중단 재개와 구분한다.

고수준 `lock_family`, `run_ordered_pilot`, `campaign`, `postfit`, `deepen` 실행 wrapper는 호출하지 않는다. 원 API를 import해 그대로 호출하며 수식·loss·공차·원 source를 복제하거나 patch하지 않는다. 원 task/scaler 검증과 비공개 failure 기록은 SHA로 고정한 기존 stdlib helper만 재사용한다.

## 자원·중단·준비 검증의 한계

ROOT는 기존 Jena4/비용/PCA6 체인이 끝난 후에만 이 묶음을 시작한다. probe는 single worker 순차, fit은 최대2 workers로 admission한다. 각 CPU2 threads/RSS3GiB, 그룹6GiB, 원 wall 종료1792222550과 C+R30GiB를 유지한다. 정밀 비용 측정은 학습·추론을 모두 끝낸 뒤 single worker 조건에서 수행한다. wrapper 자체는 한 explicit ID만 실행하고 다른 작업을 조회/종료하거나 자식을 시작하지 않는다. 기존 chain 종료 및 비용 측정과의 비중첩은 ROOT scheduler가 확인해야 하며 이 wrapper가 이를 관측했다고 주장하지 않는다.

모든 새 output과 claim은 배타적으로 생성하고 기존 run/prediction/partial을 덮거나 이어 학습하지 않는다. deadline/RSS 초과의 강제 종료는 부모 supervisor가 실제 종료와 잔존 tree를 보고해야 한다. 실패 traceback은 private cache에만 보존한다. 승인된 whole safeFIT loader가 campaign-T bytes까지 역직렬화·hash할 수 있으나 T target을 선택·평가 API로 요청하지 않는다는 범위를 명시한다.

현재 검증은 fakeTask/API와 tiny metadata를 이용한 합성 검사뿐이다. 실제 foundation의 CPU 처리량·두 worker의 메모리·전체3epoch·V 정확도·예산 내 완료는 아직 측정하지 않았다. 봉인 전 READY를 실제 실행 완료나 성능 개선으로 해석하지 않는다.
