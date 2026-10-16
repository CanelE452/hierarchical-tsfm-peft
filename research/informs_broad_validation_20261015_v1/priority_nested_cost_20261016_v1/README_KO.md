# 고정 nested PCA 비용 후속

[한국어 결과](RESULTS_WITH_COST_KO_v1.md) · [원수치](RESULTS_WITH_COST_v1.csv) · [독립 감사](INDEPENDENT_COST_REVIEW_ROOT_v1.json) · [입력 SHA](REPORT_INPUT_BINDINGS_v1.json) · [stdlib builder](build_nested_report.py)

기존 정확도3행+새 CPU 비용6개다. 새 학습0이며 전체 연구는 PARTIAL이다. 기본 builder 호출은 입력읽기·쓰기0, 명시 --execute는 감사 PASS 후 새 보고4개만 만들고 기존 출력을 거부한다.

```powershell
python -B research/informs_broad_validation_20261015_v1/priority_nested_cost_20261016_v1/build_nested_report.py --execute --public-root research/informs_broad_validation_20261015_v1
```
