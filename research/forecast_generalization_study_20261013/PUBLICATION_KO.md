# GitHub 게시 범위와 재현 안내

2026-10-13. 실험 완료 후 사용자가 이번 코드·보고서·시각자료를 main에 push하도록 명시 요청했다. 이 요청에 따라 G0–G3 연구와 평가 기준 정렬 보충 연구의 완료 자료를 함께 게시한다. 실험 당시 README·PLAN의 “commit/push하지 않았다”는 당시 상태 기록이며, 이후의 이번 게시 요청과 구분한다.

## 먼저 읽을 자료

- [G0–G3 한국어 상세 보고서](report_v1/REPORT_KO.md), [결과·코드 색인](README.md).
- [전체 방법 원수치](report_v1/all_methods.csv), [44개 핵심 학습 조건](report_v1/unique_training.csv), [단독 비용 재측정](report_v1/batch_cost.csv).
- [독립 재채점·모델 선택 검산](validation_resolved_v1/audit.md), [원자료·FIT 통계·정답·마스크·입력 경계 검산](validation_source_v1/audit.md), [최종 실험 감사](FINAL_AUDIT_V2.json).
- [보충 실험 보고서](../forecast_metric_alignment_20261013/run_v3/REPORT_KO.md).

## 포함하는 파일

실험·수치 진단·실패 경로 수리·비용 재측정·결과 병합·독립 검산·보고 생성 Python 코드, 실행별 source snapshots, JSON 결과와 실패 기록, 한국어 Markdown 문서, CSV와 PNG/SVG를 게시한다. G0–G3 보고에는 CSV11개와 PNG/SVG각5개, 별도 metric-alignment 보고에는 PNG3개가 있다. 저장소 첫 README에 이번 결과와 핵심 이미지 두 개의 색인을 추가한다.

기존 원수치·실험 소스·보고서·그림·SHA 봉인은 수정하지 않았다. 최종 [REPORT_MANIFEST](report_v1/REPORT_MANIFEST.json)과 [FINAL_AUDIT_V2](FINAL_AUDIT_V2.json)는 실험 완료 시점의 감사다. 게시 이후 HEAD·새 README 색인·이 문서를 포함한 감사로 바꾸어 쓰지 않는다. 이전 보고 metadata seal과 실패한 실행도 기록으로 보존한다.

`run_resolved_v1/RESULT.json`은 최종 과학 결과의 정본이다. `run_full_v1`의 실패4경로는 `run_repair_v1`에서 별도로 재실행했으며, 두 실행과 clean-cost 결과를 SHA로 연결했다. 이름에 RUNNING이 있는 보충 JSON은 과거 진행 snapshot이다. 현재 실행 상태는 최종 RESULT 또는 보충 `run_v3/SUMMARY.json`을 따른다.

## GitHub의 링크와 이미지

독자용 문서·이미지 링크는 저장소 상대경로다. PNG/SVG를 일반 Git blob으로 포함하므로 GitHub Markdown에서 이미지를 읽을 수 있다. Windows 로컬 경로나 `.cache`를 이미지 URL로 사용하지 않는다. 저장소의 기존 `* -text` attributes를 유지해 봉인된 파일의 byte SHA를 보존한다.

큰 원본 JSON은 GitHub 웹 미리보기가 제한될 수 있으므로, 결과를 읽을 때는 Markdown 보고서와 CSV를 우선 사용한다. JSON은 Raw 또는 다운로드로 확인할 수 있다. 결과 안의 로컬 `.cache` 경로·Windows 경로는 실행 당시 자산을 지칭하는 메타데이터이며 공개 다운로드 링크가 아니다.

## 재현에 필요한 로컬 자산

보호 TEST, 원자료 NPZ, 모델 weights·checkpoint, 신규 예측 NPZ와 Python cache, 학습 stdout log는 이번 게시에 포함하지 않는다. 정확한 학습·추론 및 배열 재채점에는 다음 자산이 필요하다.

- 계약에 지정된 Jena·Robin bounded TRAIN/VAL 원자료와 관측 마스크.
- G0 재사용에 필요한 기존 `forecast_component_study_20261013` 저장 checkpoint와 재구성 진단 결과.
- pinned 로컬 Chronos-Bolt-small snapshot 및 실행 환경.
- 새 `.cache/forecast_generalization_study_20261013/`의 예측 NPZ·선택 모델과 보충 연구의 로컬 cache.

이 자산의 실제 경로와 SHA는 [시작 자산 검사](PREFLIGHT.json), 실행 RESULT와 검산 기록에 있다. 학습 source가 사용하는 기존 `forecast_component_study_20261013/component_experiment.py`와 `past_only_repr_diagnosis_20261012` 모듈은 저장소의 해당 원래 위치를 전제로 한다. source snapshot은 봉인 증거이며 snapshot 폴더에서 직접 실행하는 진입점이 아니다.

기존 실행 tag·resource ledger를 삭제하거나 덮어서 재실행하지 않는다. 정확한 재실행에는 로컬 자산 확인, 새 실행 범위와 고유 tag, 필요한 자원 여유가 필요하다. 이번 게시 작업에서는 새 학습·추론·TEST 평가를 수행하지 않는다.

## 결과의 적용 범위

236개 방법과 수치 계약을 검증했지만 반복 사용한 두 자료의 개발 VAL 진단이다. 독립 TEST 일반화, 압축의 보편적 정확도 우위, 대조학습·동적 그래프의 필요성을 입증한 결과는 아니다. Jena K=C의 MSE 개선과 MAE 악화, Robin u0 고정 회전, batch별 압축 비용 교환을 원수치 그대로 보고한다.
