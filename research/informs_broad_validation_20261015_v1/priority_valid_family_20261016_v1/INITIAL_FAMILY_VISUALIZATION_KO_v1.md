# Jena 초기 방법군과 기존 B2: 시각자료

![Jena 세 방법의 전체 V 오차 및 CPU2 warm whole batch 지연](figures/INITIAL_FAMILY_WITH_B2_REFERENCE_v1.svg)

[원 실험 보고](INITIAL_FAMILY_KO_v1.md) · [새 두 조건 원정밀도 CSV](INITIAL_FAMILY_RESULTS_v1.csv) · [기존 B2 검증 보고](B2_REFERENCE_VALIDATION_KO_v1.md) · [B2 비용 CSV](B2_REFERENCE_COSTS_v1.csv) · [그림 SHA 결속](INITIAL_FAMILY_FIGURE_BINDINGS_v1.json)

같은 Jena C21/L512/H48·seed92601·CPU2·반복 개발 V43/43344 관측 라벨의 두 오차와 B1/B8 warm whole20 median을 원 CSV에서 그대로 복사했다. B2 정확도는 기존 fullV 재사용이고 새 CPU6/비용이 연결되었다. LoRA R8와 B4 K6는 실제 신규 완료된 학습/추론/비용이다. p90·CI·seed 분산은 그리지 않았다. 모든 축은0부터이며 표시 숫자만 반올림한다. B8은8개 입력 한 batch의 전체 초 단위 시간이다.

세 방법 search3/refit1의 update·exposure 동일. B2/B4 frozen F, LoRA effective F 적응. B2 원 학습/정규화 ancestry 유지 · 한 seed92601/반복 개발 V · 수렴·독립 일반화 미확정. LORA_R8_C21: selected refit 1.0 epoch / 12020 exposures; B4_K6: selected refit 1.0 epoch / 12020 exposures. 실제 B2·LoRA·B4 모두 search1863/refit752 update·29769/12020 exposure다. LoRA는 원 C21의 effective F를 적응하고 merge하며 B2/B4의 frozen F 경로와 다르다.

모델 로드·LoRA merge·앞서 생성한 F pool cold load는 warm whole timer에 포함되지 않는다. cold-start 전체 latency나 checkpoint 비용 비교가 아니다. B2 원4GiB 학습 guard/정규화 ancestry를 새3GiB 학습으로 소급 변경하지 않는다. F-only resident bytes는 별도 기록 없음(NA)이며 전체 Route weights/RSS로 대신하지 않는다.

이 그림은 동일 표현·동일 decoder·동등 compute의 효과 분리가 아니다. 선택 refit1 및 update/exposure는 같고 방법·모수·K/F width·학습률은 다르다. 검사 통과를 우위·수렴·독립 일반화 또는 치료 효과로 해석하지 않는다. 전체 MASTER 미완료다. 원 MD 보고가 먼저이며 그림 실패는 다른 독립 경로 중단 gate가 아니다. 새 통계/trial 재집계·모델/NPZ/CP/Task·학습·추론·비용·프로세스 제어·Git·network는0이다. 공개 SVG와 private PNG를 만들고 입력 전후 SHA 및 실제 artist12개 값을 대사했다.
