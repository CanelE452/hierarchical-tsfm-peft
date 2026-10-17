# 검증한 재현 경로

명령의 repository working directory는 clone root다. 기본 실행은 no-op이다.

```text
python -m research.adapter_diagnostics_20261017_v2.run
python -m research.adapter_diagnostics_20261017_v2.run --mode audit --asset-root ASSET_ROOT --cache-root PRIVATE_CACHE
python -m research.adapter_diagnostics_20261017_v2.run --mode plan --asset-root ASSET_ROOT --cache-root PRIVATE_CACHE
python -m research.adapter_diagnostics_20261017_v2.run --mode execute --plan REPORT_ROOT/EXPERIMENT_PLAN.json --asset-root ASSET_ROOT --cache-root PRIVATE_CACHE
python -m research.adapter_diagnostics_20261017_v2.run --mode report --asset-root ASSET_ROOT --cache-root PRIVATE_CACHE
python -m research.adapter_diagnostics_20261017_v2.run --mode verify --asset-root ASSET_ROOT --cache-root PRIVATE_CACHE
python -m research.adapter_diagnostics_20261017_v2.reporting --reaggregate REPORT_ROOT
```
실제 실행 및 종료코드는 VALIDATION.json과 root 실행 영수증을 따른다. plan/execute는 봉인 source/data identity가 맞는 승인 private 자산을 필요로 한다. SOURCE_BINDINGS.json은 수치 source SHA를, requirements-lock.txt와 START_STATE는 환경·출발 commit을 남긴다.

마지막 --reaggregate는 공개 ERROR_BY_CHANNEL의 SSE/SAE/count로 macro 점수를 독립 재집계하며 원자료·모델·네트워크를 쓰지 않는다. --figures REPORT_ROOT는 공개 집계 CSV만으로 F1~F5를 재생성한다. F6 curve 원수치는 private에만 있으며 승인된 해당 hash의 private prediction이 있어야 정확히 재생성할 수 있다. main clone만으로 원자료까지 완전 재현한다고 주장하지 않는다.

공개 stdlib 검산 helper의 실제 인수는 --repo와 --cache-root다.

```text
python -B reports/adapter_diagnostics_20261017_v2/code/verify_a03_snapshot.py --repo . --cache-root PRIVATE_CACHE
```

[검산 코드](code/verify_a03_snapshot.py)는 공개 plan/score/receipt JSON으로 cohort·진행량·지문·FP64 집계를 확인하고 PRIVATE_CACHE에 검증 JSON을 쓴다. 원자료·예측 NPZ·checkpoint를 로드하지 않으며, private 실물의 지문 재검증은 G7의 별도 결속 경로다. [기록된 A03 검산 영수증](A03_INDEPENDENT_COHORT_CHECK.json): status=PASS, exit_code=0, endpoint=72, public artifact SHA=144, CP0 group=6, parity PASS=72. 원 private 예측값은 공개하지 않는다.


E04 보완 실행은 원 106-job 실행기와 별도로 수행했다. 공개 worker는 [lora_cost_runner.py](code/lora_cost_runner.py)와 [lora_cost_runner_v2.py](code/lora_cost_runner_v2.py)의 execute_job(job, ctx)이며, 승인 private 자산·원 누적 CAMPAIGN 장부·인증 GPU parent lease가 필요하다. 두 private controller의 SHA는 확장 계획과 receipt의 execution_control에 결속되어 있다. 실제 실행 명령은 아래와 같다.

```text
python PRIVATE_CACHE/e04_orchestrate.py --execute
python PRIVATE_CACHE/e04_parent_v2_orchestrate.py --execute
```

첫 controller는 Jena parent convention 실패와 PeMS 성공을 모두 보존했다. v2는 Jena만 교정했으며 PeMS를 다시 측정하지 않았다. 초기 manifest의 artifact/state 2-field label과 deployment의 scaler 포함 3-field parent를 각각 exact 검증한다. 기존 106개 job과 원예산은 변하지 않았다. 위 generic run --mode execute는 최초106개 계획을 실행한 역사적 명령이며, 확장 E04 재실행은 별도 worker/parent lease 경로를 따른다.

실제로 검증한 모델 로드 없는 자산 검사:

```text
python reports/adapter_diagnostics_20261017_v2/code/lora_cost_runner.py --inspect --cohort jena --asset-root ASSET_ROOT --cache-root PRIVATE_CACHE
python reports/adapter_diagnostics_20261017_v2/code/lora_cost_runner.py --inspect --cohort pems08 --asset-root ASSET_ROOT --cache-root PRIVATE_CACHE
python reports/adapter_diagnostics_20261017_v2/code/lora_cost_runner_v2.py --inspect --cohort jena --asset-root ASSET_ROOT --cache-root PRIVATE_CACHE
```

보조 paired-origin/band/trajectory gate는 reporting.verify가 code/verify_report_supplements.py를 호출하여 실제 공개 집계와 private SHA를 대조한다. CODE와 report/code의 최종 테스트 대상 SHA는 TEST_VALIDATION.json에 있다.


최종 공개 명령 재현의 실제 exit code·시각·stdout SHA·집계 CSV SHA·그림 SHA는 [PUBLIC_REPRODUCTION_VALIDATION.json](PUBLIC_REPRODUCTION_VALIDATION.json)에 있다. B03의 native-FP32 엄격 검산은 캠페인의 Python3.11.16/NumPy2.4.6/OpenBLAS0.3.31·BLAS2 환경을 고정한다. [다른 NumPy/BLAS 환경 대사](TRAJECTORY_ENVIRONMENT_KO.md)에서 MKL의 FP32 reduction 차이와 두 환경의 동일 FP64 결과를 구분했다. 기본 python의 버전이 같다고 가정하지 않는다.
