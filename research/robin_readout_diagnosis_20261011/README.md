# Robin 원채널 readout 진단

[상세 한국어 보고서](REPORT_KO.md) · [사전 계획](PLAN.md) · [실행 기록](EXECUTION_NOTE_KO.md)

같은 저장 F 예측에서7개 readout 조건을 비교했다. 기존 D8 손상은 residual prior/diagonal support로 대부분 없어졌지만 Raw F 대비 MSE 이익은1% 미만이고 모든 선택 디코더의 MAE는 더 높았다. [수치](result.json), [독립 검증](verification.json), [최종 감사](final_audit.json)를 함께 확인한다. 한 자료·반복 VAL·conditional TRAIN CV의 진단이며 TEST 결과가 아니다.

![Readout comparison](figures/01_comparison.png)

코드·보고서·PNG/PDF는 GitHub 문서 기준 상대경로로 연결했다. [게시 범위와 재현 한계](PUBLICATION_KO.md), [게시 검사](publication_checks.json)를 함께 확인한다. 전체 실험·배열 검증에는 미게시 원본/예측 cache와 원래 보존 snapshot이 필요하다.
