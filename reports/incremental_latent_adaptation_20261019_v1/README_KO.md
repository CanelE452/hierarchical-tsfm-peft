# 고정 G 이후 잠재 잔차 적응

실제 공개 입력에서 재생성한 결과다. 고정된 독립 수치 검산 상태: **PASS**. 최종 완료 상태는 [VALIDATION_REPORT.json](VALIDATION_REPORT.json)을 확인한다.
필수 비교는 G/기존 alpha/출력부만/F/no-F 잠재/원본 잔차 MLP와 선형 프로브다. 음의 자료·seed를 제외하지 않는다.
자료별 F 대 기존 fitted alpha의 평균 MSE 차이(F−alpha): jena 0.0692452; pems32 0.00194507. 양수이면 기존 alpha가 더 정확하다. 순차 블록의 평균 우위도 확보하지 못했다.
실제 모델 수·내부 단계 수·조건부 미진입·미완료 여부는 [EXECUTION_STATUS.csv](EXECUTION_STATUS.csv)와 [RESUME_QUEUE.md](RESUME_QUEUE.md), 원격 게시 증거는 [PUBLICATION_VERIFICATION.json](PUBLICATION_VERIFICATION.json)을 따른다. 이 문서는 실험 검산과 원격 게시 검증을 별도로 연결한다.

- [전체 분석](RESULT_ANALYSIS_KO.md)
- [주장 상태](CLAIM_STATUS.md)
- [부모·자료 계약](PARENT_AND_DATA_CONTRACT.json)
- [고정 수치 검산](NUMERICAL_VALIDATION.json)
- [최종 검증](VALIDATION_REPORT.json)
- [그림·공개 원점](FIGURE_MANIFEST.json)
- [실측 값](FIGURE_POINTS.csv)
- [재현](REPRODUCE.md)

V는 반복 개발 평가다. 두 seed 범위는 신뢰구간이 아니다. 보호 T source-array 숫자 접근은 없었으며, 역사 문서의 우연한 숫자 노출·상류 전처리·사전학습 중복은 INFORMATION_ACCESS_AUDIT의 한계를 따른다.
