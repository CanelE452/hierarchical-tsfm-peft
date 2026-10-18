# 보고 집계의 방법과 출처

공식 RevIN 원문·코드 revision·라이선스와 포트 차이는 별도 [METHOD_REFERENCES.md](METHOD_REFERENCES.md)를 따른다.

기존 B2와 데이터·학습 계약은 봉인된 broad/diagnostics/improvements source를 재사용한다. 이번 변경은 원 C채널 창별 outer RevIN이다. 공식 전체 학습 recipe의 재현으로 부르지 않는다.

```text
공식 원문·수정점·source pin은 CAMPAIGN_CONTRACT.json과 기존 METHOD_REFERENCES.md를 따른다.
```

RevIN은 causal fill와 train z-score 뒤 biased variance, eps1e-5, detached mean/std, gamma1/beta0를 사용한다. 공식 affine inverse 분모 gamma+eps²를 포함한다. anchor까지 포함한 출력을 한 번 inverse하고 global-normalized target 공간에서 loss를 구한다. 실제 source hash·수정점은 CAMPAIGN_CONTRACT.json 및 RUNS의 bindings를 따른다.

기존 검증 결과·그림의 재사용 경로와 원 SHA는 아래 목록이다. 과거 측정 조건은 그대로 유지하며 새 RevIN 결과로 재레이블하지 않는다.

```json
{
  "artifacts": [
    {
      "reference": "repo://reports/adapter_diagnostics_20261017_v2/figures/accuracy_cost.svg",
      "scope": "Historical accuracy-cost; multiple backbones and CPU/CUDA continuation scope",
      "sha256": "ddc925f7b288c78112f65ffd18a8cab429661082ab2de6e16af25325bcc507b0",
      "url": "https://github.com/CanelE452/hierarchical-tsfm-peft/blob/e2e85acdaffa12c4625d4c66c7c117066e15a295/reports/adapter_diagnostics_20261017_v2/figures/accuracy_cost.svg"
    },
    {
      "reference": "repo://reports/adapter_diagnostics_20261017_v2/figures/learning_curves.svg",
      "scope": "Historical capped training and selection context",
      "sha256": "f7818f8590608ed7ad35b9f5719e768cd591e57b8dceb4804943d1df46a2693f",
      "url": "https://github.com/CanelE452/hierarchical-tsfm-peft/blob/e2e85acdaffa12c4625d4c66c7c117066e15a295/reports/adapter_diagnostics_20261017_v2/figures/learning_curves.svg"
    },
    {
      "reference": "repo://reports/adapter_diagnostics_20261017_v2/FIGURE_METADATA.json",
      "scope": "Original figure metadata and interpretation",
      "sha256": "93fbb9141125f3d24115e2f4283519fafafc4132209cad1951186e7602c41f44",
      "url": "https://github.com/CanelE452/hierarchical-tsfm-peft/blob/e2e85acdaffa12c4625d4c66c7c117066e15a295/reports/adapter_diagnostics_20261017_v2/FIGURE_METADATA.json"
    },
    {
      "reference": "repo://reports/adapter_diagnostics_20261017_v2/SOURCE_BINDINGS.json",
      "scope": "Original saved prediction and source bindings",
      "sha256": "078131a8f3033fc2e3bc3576b04d309515ec7be65b115d4981a233e28ff39214",
      "url": "https://github.com/CanelE452/hierarchical-tsfm-peft/blob/e2e85acdaffa12c4625d4c66c7c117066e15a295/reports/adapter_diagnostics_20261017_v2/SOURCE_BINDINGS.json"
    },
    {
      "reference": "repo://reports/adapter_improvements_20261018_v1/figures/F5.svg",
      "scope": "Historical whole-path latency and prepare/fit costs",
      "sha256": "5ba9f59fa535582e0e91b945a3252ead22ee438758782e2a88ff4b4c81c3c69f",
      "url": "https://github.com/CanelE452/hierarchical-tsfm-peft/blob/e2e85acdaffa12c4625d4c66c7c117066e15a295/reports/adapter_improvements_20261018_v1/figures/F5.svg"
    },
    {
      "reference": "repo://reports/adapter_improvements_20261018_v1/figures/F5_source.csv",
      "scope": "Exact source values of historical cost figure",
      "sha256": "5d1e2c0102e66dca4275f0bc7a69e553c8109b8e944c9f5d7634826e30e93dc0",
      "url": "https://github.com/CanelE452/hierarchical-tsfm-peft/blob/e2e85acdaffa12c4625d4c66c7c117066e15a295/reports/adapter_improvements_20261018_v1/figures/F5_source.csv"
    },
    {
      "reference": "repo://reports/adapter_improvements_20261018_v1/MULTIK_AUDIT.csv",
      "scope": "Existing all-K evidence limitations; no new K training",
      "sha256": "4f26b3b8a23972246f06c0663c464106116762815b9db7f71f99bada03986731",
      "url": "https://github.com/CanelE452/hierarchical-tsfm-peft/blob/e2e85acdaffa12c4625d4c66c7c117066e15a295/reports/adapter_improvements_20261018_v1/MULTIK_AUDIT.csv"
    }
  ],
  "new_experiments": 0,
  "reuse_mode": "Pinned original links and aggregate sources; old artifacts preserved",
  "status": "VERIFIED_BYTES"
}
```
