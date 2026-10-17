# B2 개선 원인·적용 범위: 중단 시점 공개 패키지

사용자의 중지 후 이미 저장된 코드·집계 결과를 정리한 패키지다. **실험 전체 미완료**이며 재학습·재추론·새 비용 측정을 하지 않았다. 새 CAP native 점수와 기존 독립 감사 완료 결과를 구분한다.

- [한국어 결과·한계·발표 문구](STOPPED_RESULTS_KO.md)
- [모든 비교값 CSV](COMPARISONS_AT_STOP.csv) / [완료·미완료 상태 CSV](STATUS_AT_STOP.csv)
- [공개 입력·생성 코드 SHA 및 점검](STOPPED_SOURCE_BINDINGS.json)
- [정지 상태 공개 영수증](STOP_STATUS_PUBLIC.json)
- [기존 오류 분해 상세·그림](ERROR_DECOMPOSITION_KO.md)
- [보고 생성 코드](code/build_stopped_report.py)

![선택 모델 결과](figures/STOPPED_SELECTED_MODELS.png)

![공통 학습량 비교와 미완료](figures/STOPPED_MATCHED_UPDATES.png)

![단계별 완료 상태](figures/STOPPED_WORK_STATUS.png)

`SAME_UPDATE`는 A_fit 검색 checkpoint, `SELECTED_A60`는 A_stop 선택 후 fresh A60 refit이다. CAP의 선택 모델 점수를 공통 update의 빈칸에 넣지 않는다. 독립 미래 일반화와 수렴은 미확정이며 보호 T는 잠금 유지다. 재현 명령과 계약은 상세 보고서에 있다. 원자료·예측 배열·체크포인트·개인 절대경로는 이 패키지에 포함하지 않는다.
