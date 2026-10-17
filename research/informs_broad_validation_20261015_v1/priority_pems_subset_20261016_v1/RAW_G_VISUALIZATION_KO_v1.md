# PeMS08 C32 Raw·직접 G 시각자료

![정규화 정확도와 CPU2 warm whole 비용](figures/RAW_G_FOUR_PANEL_v1.svg)

[실험 보고](RAW_G_KO_v1.md) · [원정밀도 CSV](RAW_G_RESULTS_v1.csv) · [그림 결속](RAW_G_FIGURE_BINDINGS_v1.json) · [코드](plot_raw_g_v2.py) · [같은 독립 감사](INDEPENDENT_RAW_G_AUDIT_ROOT_v1.json)

감사 상태: `PASS`. Raw/G 두 자리를 고정하되 실제 검증된 보고 행만 bar로 그렸다. 누락된 peer나 실패한 Raw batch 비용은 NA로 표시하며 가짜 0이나 완료 행을 만들지 않는다.

MSE가 주지표이며 MAE와 CPU2 B1/B8 ordinary whole20 median을 함께 표시했다. B8은 8개 입력 batch 전체의 초이고 8로 나누지 않았다. 표시 라벨만 반올림했으며 모든 실제 bar 높이와 상태는 canonical CSV 원 수치와 exact 대사했다. CI·seed 분산·새 p90을 추가하지 않았다.

C32 / L512 / H12 / V148 / seed92601 · 반복 개발 V · 제한 search와 선택 refit: 수렴·독립 일반화 미확정. Raw와 G는 구조·학습량이 다름. B2·압축 가치·Bolt 필요성의 충분한 검증이 아님. NA는 미완료이며 0이 아님.

Raw는 동결 Chronos-Bolt C32와 optimizer0, G는 F 없이 채널 공유 512→32→12 GELU innovation·현재값 anchor다. 구조·학습량이 다르므로 이 그림을 B2·압축의 가치·Bolt 필요성의 충분한 증거로 확대하지 않는다. 실제 선택 refit 길이와 학습 상태는 원 표를 따른다. upstream 미래 보간·전처리 계보는 UNKNOWN이다.

warm whole timer는 cold model load를 제외한다. 그림에서 CPU parameter+buffer resident·RSS·F 단독 resident를 대입하거나 새 측정하지 않는다. canonical audit와 보고의 입력 SHA를 재검사했고 원수치/trial 재집계·Task/원자료/NPZ/CP/모델 해석·학습·추론·비용 재측정·프로세스 제어·Git·network는 하지 않았다. 공개 SVG와 private cache PNG를 별도로 만들었다. 원 보고서가 먼저이며 그림 실패는 다른 실험의 중단 gate가 아니다. 전체 MASTER와 원 subset 전체 실험은 미완료다.
