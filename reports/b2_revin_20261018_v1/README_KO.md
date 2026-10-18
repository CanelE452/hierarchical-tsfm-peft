# B2 + RevIN 최소 실행 결과

정확도–비용, 대응 RevIN 비교, Jena 구성 기여·한계의 세 묶음이다. 신규 개선 방법은 outer RevIN 하나이며 두 데이터의 seed92601/92602를 유지한다.

[결과 분석](RESULT_ANALYSIS_KO.md) · [선택 결정](DECISIONS_KO.md) · [재현](REPRODUCE_KO.md)

- [T1 정확도–비용](T1_ACCURACY_COST.csv)
- [T2 RevIN cap3/6 대응](T2_REVIN_MATCHED.csv)
- [T3 Jena 구성 기여](T3_COMPOSITION_LIMITS.csv)

V는 반복 개발 평가로 이미 사용한 자료이며 독립 테스트 결과가 아니다. 새 선택에 M_cal/T를 사용하지 않았으며, 기존 T 노출과 pretraining overlap UNKNOWN의 범위는 원 계약을 따른다. 두 seed·고정 K의 결과를 모든 데이터·압축률의 일반화로 확대하지 않는다.

그림: [F1](figures/F1_accuracy_cost.png), [F2](figures/F2_revin_paired.png), [F3](figures/F3_channel_horizon.png), [F4](figures/F4_astop_curves.png). SVG와 source CSV도 같은 figures 폴더에 있다. 실제 검산과 원격 게시 증거는 별도 검증 JSON을 따른다.

[accuracy_cost.svg](../../reports/adapter_diagnostics_20261017_v2/figures/accuracy_cost.svg) — Historical accuracy-cost; multiple backbones and CPU/CUDA continuation scope；SHA256 `ddc925f7b288c78112f65ffd18a8cab429661082ab2de6e16af25325bcc507b0`

[learning_curves.svg](../../reports/adapter_diagnostics_20261017_v2/figures/learning_curves.svg) — Historical capped training and selection context；SHA256 `f7818f8590608ed7ad35b9f5719e768cd591e57b8dceb4804943d1df46a2693f`

[FIGURE_METADATA.json](../../reports/adapter_diagnostics_20261017_v2/FIGURE_METADATA.json) — Original figure metadata and interpretation；SHA256 `93fbb9141125f3d24115e2f4283519fafafc4132209cad1951186e7602c41f44`

[SOURCE_BINDINGS.json](../../reports/adapter_diagnostics_20261017_v2/SOURCE_BINDINGS.json) — Original saved prediction and source bindings；SHA256 `078131a8f3033fc2e3bc3576b04d309515ec7be65b115d4981a233e28ff39214`

[F5.svg](../../reports/adapter_improvements_20261018_v1/figures/F5.svg) — Historical whole-path latency and prepare/fit costs；SHA256 `5ba9f59fa535582e0e91b945a3252ead22ee438758782e2a88ff4b4c81c3c69f`

[F5_source.csv](../../reports/adapter_improvements_20261018_v1/figures/F5_source.csv) — Exact source values of historical cost figure；SHA256 `5d1e2c0102e66dca4275f0bc7a69e553c8109b8e944c9f5d7634826e30e93dc0`

[MULTIK_AUDIT.csv](../../reports/adapter_improvements_20261018_v1/MULTIK_AUDIT.csv) — Existing all-K evidence limitations; no new K training；SHA256 `4f26b3b8a23972246f06c0663c464106116762815b9db7f71f99bada03986731`
