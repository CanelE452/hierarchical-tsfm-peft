# B2 우선 검증: Jena 핵심 비교 완료, 반복 확인은 후속

**B2 계산 경로는 통과했고 B0·두 no-Bolt 선형 대조보다 오차가 낮았다. Ridge·G·제한 학습 LoRA보다 정확하지 않았으며, no-Bolt 대비 CPU 추론 비용도 크게 높았다.** 단일 seed의 반복 개발 V 결과다. 방법론의 일반적 우위나 전체 연구 완료를 주장하지 않는다.

- [판단 보고: 구조·추가 가치·Bolt 필요성·정확도와 비용·후속 범위](INFORMS_METHOD_STATUS_KO.md)
- [원정밀도 11조건 결과 CSV](PRIORITY_RESULTS.csv), [방법 수식과 두 직접 대조의 차이](METHOD_CONTRACT.md)
- [실제 중단 작업·체크포인트·재개 한계](PAUSE_AND_RESUME.md), [우선순위 큐 snapshot](PRIORITY_QUEUE.json)
- [실제 P0 결과](diagnostics/p0_root_v1/RESULT.json), [실행 종료·자원 반환 확인](EXECUTION_CLOSURE.json)
- [R5 저장 통계·비용 독립 대사](INDEPENDENT_R5_REVIEW.json), [PCA/LoRA 재추론·비용 독립 대사](INDEPENDENT_LEGACY_REVIEW.json)
- [사전 봉인 계획](PREREG_PLAN.json), [PCA/LoRA 추가 측정 계약](LEGACY_BASELINE_SPEC.json), [보고 입력 결속](REPORT_INPUT_BINDINGS.json)

실험 코드는 [직접 head](direct_no_bolt.py), [과거 prefix 로더](safe_jena_prefix.py), [학습·전체 V·검산·재추론·비용 runner](priority_runner.py), [P0](check_priority_path.py), [기존 PCA/LoRA CPU 연결](legacy_baseline_cpu.py)이다. 각 기본 CLI는 실험 입력을 읽지 않는다. 실제 실행에는 명시적인 `--execute`, 로컬 자산, 봉인된 계약과 원 자원 상한의 외부 supervisor가 필요하다.

R5a는 동일한 최종 P/E/D를 고정하고 전체512×6 과거를 사용한다. R5b는 같은 구조를 no-Bolt 예측에 맞춰 별도 학습한다. 기존 정확도·완료된 학습은 재사용했고, 빠진 CPU 재추론·비용만 추가했다. 본문 표는 표시 자릿수를 줄였으며 CSV가 원정밀도 값이다.

Raw 비용 실패·LoRA .25epoch 제한·수렴 미확정·새 독립 평가 부재를 보존한다. 새 보호 T 접근·PPT/PDF·새 그림·대규모 포장은 수행하지 않았다. PeMS08과 추가 seed의 최소 후속은 별도 계약·실행 결과로 구분한다. 원 데이터·예측 NPZ·체크포인트는 공개하지 않으며 이 공개 자료만으로 학습 전체가 재현된다고 주장하지 않는다.
