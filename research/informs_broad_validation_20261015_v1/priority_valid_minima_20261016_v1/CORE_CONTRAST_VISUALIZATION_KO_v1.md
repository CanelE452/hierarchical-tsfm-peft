# Beijing core contrast 시각자료

![B0 K6와 B2 K6/K11의 전체 V 오차 및 실제 CPU2 whole median](figures/CORE_CONTRAST_v1.svg)

[원정밀도 CSV](CORE_CONTRAST_RESULTS_v1.csv) · [실험 보고](CORE_CONTRAST_KO_v1.md) · [보고 입력 SHA](CORE_CONTRAST_BINDINGS_v1.json) · [실제 감사](INDEPENDENT_CORE_CONTRAST_AUDIT_ROOT_v1.json) · [그림 결속](CORE_CONTRAST_FIGURE_BINDINGS_v1.json) · [코드](plot_core_contrast.py)

같은 Beijing C11/L512/H48·CPU2·V43/20539 observed labels·seed92601의 세 경로를 표시한다. B2 K6는 기존 완료 결과를 재사용했고 B0 K6·B2 K11은 새 완료 결과다. 모든 축은0부터이며 표시 숫자만 반올림한다. CSV 원정밀도·12개 실제 artist 값은 변경하지 않았다. 지연시간은 ordinary whole20 median의 초/batch이며 B8은8 origins 한 batch 전체다. p90·seed 분산·신뢰구간 표시는 추가하지 않았다.

P/E/D는 미래 forecast MSE 학습 · K6 PCA 초기화 / K11 identity 초기화 · P/K·capacity·선택 refit 길이가 함께 다름. 순수 압축 효과로 해석하지 않음. 단일 seed92601 · 반복 개발 V · 제한 학습: 수렴·독립 일반화 미확정.

이 그림은 같은 decoder만의 비교나 같은 가중치 표현 비교가 아니다. B0/B2 모두 frozen Bolt를 사용하며 새 noBolt 실험이나 Bolt의 일반적 필요성을 증명하지 않는다. 오차·전체 비용·학습량/용량 차이를 같이 보며 검사 통과를 성능 개선이나 독립 일반화로 바꾸어 말하지 않는다. 전체 MASTER 미완료다.

원 CSV/실제 audit/report SHA에 결속하고 입력 전후 SHA·실제 artist 값을 대사했다. 새 통계/trial 재집계·NPZ/CP/model 읽기·학습·추론·비용·process 제어·Git·network는0이다. SVG/이 MD/결속 JSON은 공개용이고 PNG preview는 private다.
