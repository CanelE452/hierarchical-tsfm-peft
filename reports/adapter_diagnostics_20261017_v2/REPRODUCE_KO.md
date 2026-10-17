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
