# 재개 상태

완료 수치는 보존한다. 점수에 따라 후보·방법군·선택규칙을 바꾸지 않는다.

등록된 단계는 완료됐다. cap에서 개선 중인 모델의 수렴은 미확정이며 추가 예산 실험은 별도 승인·등록 범위다.

기존 상태를 보존하며 같은 실행기를 재개하는 명령:

```text
python -m research.raw_context_multiscale_20261018_v1.run execute --asset-root <local-asset-root> --plan reports/raw_context_multiscale_20261018_v1/EXPERIMENT_PLAN.json
```

예산 부족으로 미입장한 연장은 기존 immutable decision을 덮어써서 수행하지 않는다. 추가 예산·새 attempt를 등록한 후 진행한다.
