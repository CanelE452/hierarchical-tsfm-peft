# 게시 범위와 재현 안내

2026-10-13. 사용자의 이번 명시적 요청에 따라 완료된 FORECAST 진단·예측 성분 연구의 코드, 결과, 한국어 Markdown 보고서, CSV와 PNG/SVG를 main에 게시한다. 실험 당시 기록의 “자동 commit/push 없음”은 실험 완료 시점의 상태이며, 이후의 명시적 게시 요청과 구분한다.

## 먼저 읽을 자료

- [현재 확인된 것과 다음에 검증할 질문](STATUS_AND_NEXT_KO.md)
- [전체 실험 보고서](report_v1/REPORT_KO.md), [코드·결과 색인](README.md)
- [고정 과거 창의 재구성 진단 최종 해석](reconstruction_audit_v1/REPORT_KO_v2.md)
- [논문·교수님 제안에 대한 비판적 검토](DISCUSSION_KO.md)

## 포함하는 자료

새 연구 폴더의 실험·검산·보고서 생성 Python 코드, Stage A/A2/B와 SELF33의 JSON 결과, 재구성 추가 진단, 독립 검산과 원수치 CSV, 연구·보고 문서, PNG/SVG 그림을 포함한다. 저장소 첫 README에 결과 색인을 추가하고 이번 날짜의 작업 기록을 함께 게시한다.

그림과 문서 링크는 저장소 상대경로다. 이미지 파일은 일반 Git blob으로 게시한다. 기존 `* -text` attributes로 봉인된 소스·결과의 byte SHA를 보존한다. 원본 JSON의 `.cache` 경로는 당시 로컬 자산을 가리키는 메타데이터이며 공개 파일 다운로드 링크가 아니다.

큰 JSON은 GitHub 웹 미리보기가 제한될 수 있으므로, 결과는 통합 Markdown 보고서와 CSV에서 읽고 JSON 원본이 필요하면 Raw 또는 다운로드를 사용한다.

중간 실패한 검산 v1과 이후 성공한 v2는 모두 보존했다. 최종 과학 결과는 [validation_final_v2](validation_final_v2.json), [보고서 수치 검산](REPORT_VERIFICATION_v1.json), [재구성 독립 검산](RECONSTRUCTION_AUDIT_VERIFICATION_v1.json)을 따른다. `FINAL_AUDIT.json`은 실험 완료 당시의 감사이며, 게시 이후 HEAD나 새 요약 문서를 포함한 감사로 바꾸어 쓰지 않는다.

## 재현에 필요한 로컬 자산

정확한 학습·추론 재실행에는 기존 bounded TRAIN/VAL 원자료, Stage A/A2의 이전 FORECAST 초기 checkpoint, 로컬 pinned Chronos-Bolt-small checkpoint, `.cache/forecast_component_study_20261013/`의 저장 예측·모델 state가 필요하다. 데이터·모델 파일은 코드와 별개이며 이 게시에 포함하지 않는다. 보호 TEST를 새로 사용하지 않았다.

환경은 Python3.11, PyTorch2.10.0+cu128, RTX4070, float32를 사용했다. [환경·데이터 존재 확인](PREFLIGHT_v1.json), [Stage B 실제 실행 소스와 pinned 모델 기록](stage_b_full_v1/IMPLEMENTATION_AND_REPRODUCIBILITY.json)을 참조한다. 공통 loader와 이전 FORECAST 소스는 이미 저장소에 게시된 `research/past_only_repr_diagnosis_20261012/` 모듈을 사용한다.

**실행 파일은 연구 폴더 바로 아래의 원래 Python 코드다.** `stage_b_full_v1/*_snapshot.py`는 실행 당시 소스를 봉인한 감사 자료다. 경로 기준이 원래 위치를 전제로 하므로 snapshot 파일을 그 저장 위치에서 직접 실행하지 않는다.

기존 실행 tag와 결과 파일이 존재하면 재실행을 거부한다. 새 실행은 필요한 로컬 자산을 확인한 뒤 고유 tag 또는 별도 workspace를 사용한다. 이번 게시 작업은 기존 모델을 학습하거나 보호 TEST를 평가하는 작업을 다시 수행하지 않는다.

## 해석 범위

두 자료·두 seed, L512/H48, 한 K 설정과 반복 사용 VAL의 개발 진단이다. 실험 실행·수치 검산·정보 경계 검사를 완료했으며, 독립 TEST 일반화나 압축+보정의 보편적 정확도 우위를 주장하지 않는다. 후속 질문은 실행 결과와 구별하여 [다음 검증 순서](STATUS_AND_NEXT_KO.md)에 적었다.
