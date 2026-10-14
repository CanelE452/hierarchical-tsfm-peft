# Matched-readout 완료 결과 main 게시

[최신 보고서·그림 6개](reports/matched_REPORT_KO.md) · [현재 판단과 다음 질문](reports/interpretation_REPORT_KO.md) · [이전 완료 단계 전체](../publication_main_20261014_v1/README_KO.md)

RAW33C와 E512K의 신규 8 fits·56 ridge 후보, FIT/TUNE 후보 예측112개, 선택 예측36개 및 기존 E33K N2 계수4개 재사용을 완료했습니다. 모든 E/D/F는 새로 load·forward·update하지 않았으며, cached 과거 E와 원 과거 C만 사용했습니다.

```text
두 seed 평균 정규화 MSE     Robin        Beijing
RAW-F                      0.34040189   0.58957541
MAIN                       0.56954654   0.66195720
E33K                       0.92409283   0.73662621
RAW33C                     0.69578854   0.74559667
E512K                      0.90270308   0.75732091
```

시간 요약을 없앤 E512K는 Robin에서 소폭 좋아지고 Beijing에서 나빠졌습니다. RAW33C도 RAW-F/MAIN 격차를 해결하지 못했습니다. 입력 차원·계수 수가 다르므로 encoder 단독 효과를 확정하지 않습니다. 반복 VAL30·노출 EVAL146·두 E seed이며 독립 일반화 검증이 아닙니다.

![실제 평가 오차](../report_matched_readout_v2/figures/01_eval_seed_dots.png)

![입력·계수 용량](../report_matched_readout_v2/figures/04_capacity_rank.png)

독립 저장 배열 감사 v2는 PASS/실패0, 최종 파일·링크·CSV 감사는726/0입니다. 이전 감사 v1의 shape/count 오류와 실패 기록도 보존합니다. E/F 순전파·SVD 최적성·새 배포 지연을 독립 재실행한 검사는 아닙니다.

[전체 원수치 CSV](../report_matched_readout_v2/METRICS.csv) · [56 후보](../report_matched_readout_v2/ALPHA_GRID.csv) · [독립 감사](../validation_v1/MATCHED_READOUT_AUDIT_v2_ACTUAL.json) · [최종 726 검사](../validation_v1/MATCHED_REPORT_FINAL_VERIFICATION_v3.json)

실험·검산 코드, 고정 조건, 실제 producer JSON 사본·13 terminal resource receipt, CSV와 PNG/SVG를 게시합니다. NPZ·원데이터·모델·checkpoint와 새 MAIN residual 실험의 준비/실행 소스는 이 snapshot에 포함하지 않습니다. JSON 내부의 캐시·Windows 경로는 로컬 재현 계보이며 GitHub에 존재하는 파일 링크가 아닙니다. 연구 정확도 목표는 미해결이며 다음 승인 범위의 실행은 계속됩니다.

[게시 파일·SHA](FILES.json) · [링크 확인](LINK_AUDIT.json) · [원 보고서 SHA](ORIGINAL_REPORTS.json)
