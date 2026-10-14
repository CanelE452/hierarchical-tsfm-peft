# 완료된 연구 결과 main 게시 색인

게시 범위는 Bi-Hash 미래 정렬, 작은 K 예측 목적 정렬, D0–D3 진단, joint E/D·Appliances 손실, RF1/RF3, E/F/D 분리 진단까지입니다. 새 matched-readout 단계는 승인 후 실행 준비 중이며 이번 코드·결과 snapshot에 포함하지 않습니다.

**연구 목적이 해결됐다는 뜻은 아닙니다.** 현재 작은 K 경로는 RAW 대비 정확도 격차가 남고, 반복 개발 VAL·노출 EVAL·소수 seed의 제한이 있습니다. Oracle는 실제 미래를 쓰는 진단이며 배포 방법이 아닙니다. ExoTimer 전체 문장·LLM 시스템을 재현한 결과가 아닙니다.

## 보고서와 근거

- [Bi-Hash 미래 정렬 전체 결과](reports/bihash_REPORT_KO.md) · [봉인 원본](../../bihash_future_prediction_20261013/report_canonical_final_v2/REPORT_KO.md)
- [Bi-Hash 파일럿 기록](reports/bihash_pilot_REPORT_KO.md) · [봉인 원본](../../bihash_future_alignment_20261013/report_v1/REPORT_KO.md)
- [작은 K 예측 목적 정렬·새 자료 비교](reports/alignment_REPORT_KO.md) · [봉인 원본](../../smallk_predictive_alignment_20261013/report_v2/REPORT_KO.md)
- [D0–D3 표현·예측기·복원 진단](reports/diagnostics_REPORT_KO.md) · [봉인 원본](../report_followup_v3/REPORT_KO.md)
- [Joint E/D 목표 및 Appliances 손실 비교](reports/joint_REPORT_KO.md) · [봉인 원본](../report_joint_followup_v2/REPORT_KO.md)
- [RF1/RF3 디코더·균형 순서 비용 비교](reports/sequence_REPORT_KO.md) · [봉인 원본](../report_sequence_followup_v1/REPORT_KO.md)
- [E/F/D 분리·native ridge·Oracle 진단](reports/stage_REPORT_KO.md) · [봉인 원본](../report_stage_separation_v2/REPORT_KO.md)

[현재 해석과 다음 질문](reports/interpretation_REPORT_KO.md)

## 최신 핵심 결과

```text
두 seed 평균 정규화 MSE       Robin       Beijing
RAW                         0.340402    0.589575
MAIN                        0.569547    0.661957
N1: 미래 latent 예측 + D    0.914009    0.727793
N2: 과거 E → 미래 C         0.924093    0.736626
N3: 고정 N1 + D 보정        0.946790    0.737978
```

N2는 persistence·FIT 평균 보정보다 좋지만 RAW·MAIN보다 나쁩니다. 현재 hybrid native B1 비용은 약2.66–3.93ms, 같은 측정의 RAW는48.30–51.64ms입니다. 정확도 목표는 미충족이며 이전 측정과 RAW 절대지연이 달라 안정적인 범용 속도 우위를 주장하지 않습니다.

![최신 deploy 오차](../report_stage_separation_v2/figures/01_deploy_mse_mae_dots.png)

![실제 whole 배포 비용](../report_stage_separation_v2/figures/06_b1_b8_paired_cost.png)

## 게시·재현 범위

실험·검산 소스, 고정 조건, 최신 보고서·CSV·PNG/SVG, 선택한 terminal/실패 감사와 resource receipt를 게시합니다. 원데이터, NPZ 예측 배열, 모델 가중치·checkpoint, 로그, 실행 중 matched-readout 소스와 오래된 무관한 변경은 제외합니다. 중복 대형 snapshot 전체를 게시하지 않습니다. 원 보고서/manifest/그림의 byte SHA는 바꾸지 않았습니다.

대형 JSON/CSV는 GitHub 기본 미리보기 한도를 넘을 수 있습니다. Raw 또는 다운로드로 확인할 수 있으며 이미지·Markdown 링크는 일반 Git blob입니다. JSON 내부의 `.cache`·Windows 절대경로는 원 실행 계보 기록입니다. 해당 파일이 GitHub에 있는 것으로 해석하면 안 됩니다.

[게시 파일·SHA 목록](FILES.json) · [링크 점검](LINK_AUDIT.json) · [원 보고서 SHA](ORIGINAL_REPORTS.json)
