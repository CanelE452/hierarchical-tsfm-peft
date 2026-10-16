# 고정 PCA 기준선의 별도 CPU 확대

이 폴더는 원 등록 목록에서 V 예측이 없던 고정 PCA6조건을 같은 방법의 CPU 조건으로 확인하는 별도 post-T development다. 실행 여부는 각 RESULT와 원 BASE/V receipt로 확인하며, 현재 PLAN 봉인과 합성 검사 통과는 실제 추론 완료를 뜻하지 않는다.

```text
자료          K       H
Beijing       9       48
Jena          16      48
Electricity   41,241  48
PeMS08        22,128  12
```

각 조건은 원 `PCA_FIXED_BOLT` 구현과 데이터·시점·정규화·PCA·frozen Bolt-small·원 채널 미래 MSE 정의를 유지한다. 장치는 CPU, seed92601, 입력512, row microbatch128이며 checkpoint recomputation을 사용하지 않는다. 새 ID와 별도 public/private root를 쓰고 기존 결과·중단 학습·registry·선택 잠금·점수는 보존한다.

PCA의 center/E는 buffer이며 학습 파라미터가 없다. 원 `[0,.25,1,3]` 일정은 A_stop 반복 평가이고 optimizer update·학습 exposure·학습 epoch는0이다. A50의 PCA를 확인한 뒤 원 API가 A60 PCA를 새로 만들고 전체 V를 추론한다. 원 BASE receipt의 `optimizer="AdamW"`는 고정 문자열이다. 실제 stage receipt의 `NONE_FIXED_TRANSFORM`과 업데이트0을 함께 확인하며 신경망3epoch 완료 수에 넣지 않는다.

원 safeFIT task 전체 역직렬화 계약과 legacy 정규화 이력을 유지한다. T byte가 메모리에 전혀 올라오지 않는다는 주장은 하지 않는다. 새 계산은 A50/A_stop/A60/V로 제한하고 새 T target·적합·선택·평가는 요청하지 않는다. 기존 T를 이미 본 뒤의 탐색적 결과이므로 독립적인 일반화 검증으로 해석하지 않는다.

현재 Jena4조건의 학습을 마친 뒤, CPU 비용 측정과 겹치지 않게 한 worker씩 실행한다. 기존 wall·저장 한도 안에서 실제 시간과 RSS를 확인하며 한 조건의 실패는 그 조건에 기록한다. 저장 예측 재채점과 선택된 고정 변환의 재추론도 실행 후 별도로 기록한다.

- [PLAN](PLAN.json), [등록6조건](REGISTRY.csv).
- [실행 코드](fixed_pca_runner.py), [합성 검사](test_fixed_pca_runner.py).
- [공유 stdlib 실행 유틸리티](../postT_V_only_cpu_lag_20261016_v1/cpu_lag_runner.py), [원 미완료 범위](../RESUME.md).

기본 CLI는 metadata만 확인한다. 실제 실행에는 `--execute --mode fit`, 정확한 run ID·PLAN SHA·새 diagnostic 경로·유효한 deadline·세 root와 CPU/offline 환경이 필요하다. 기존 run/prediction/claim 경로가 있으면 보존하고 새 실행을 거부한다.
