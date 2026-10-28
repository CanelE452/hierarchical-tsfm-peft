# Jena P 분기 실험과 발표 수정

사용자 승인: Jena의 시간 분기만 / 채널 분기만 실험 기본 20회와 결과를 반영한 문제정의·방법론·16·18장 정리. 추가 질문 때문에 학습을 기다리지 않는다. 전달 목표는 2026-10-28 20:00 KST다.

## 실행

- 학습 admission 마감 및 worker deadline: 2026-10-28 16:00 KST.
- 로컬 설정 봉인: `REGISTRATION_SEAL.json`; 새 commit/push는 요청되지 않아 수행하지 않았다.
- 기본 20개는 seed 92601–92610, 두 분기를 seed별로 번갈아 실행한다.
- A_stop 상한 조건에 해당하면 10개 전체의 연장 예산을 확인한다. V 값을 보고 연장하지 않는다.
- 실행 환경: `E:/CODING/proj/hierarchical-tsfm-peft/.venv/Scripts/python.exe`.
- 실행 모듈: `research.jena_p_branch_20261027_v1.controller`.
- 실제 PID와 현재 작업은 `LAUNCH.json`, `PROGRESS.json`, `.cache/jena_p_branch_20261027_v1/LIVE_WORKER.json`과 OS 프로세스를 함께 확인한다. venv launcher와 실제 Python 프로세스가 각각 보일 수 있다.
- 프로세스가 살아 있으면 중복 실행하지 않는다. 단순 관찰 타임아웃은 종료 증거가 아니다.
- 완료 결과는 선택 봉인·예측·bundle 해시로 재검증하여 재사용한다. 재개 시 같은 controller 모듈을 사용한다.
- 기술적 문제는 새 캠페인의 운영/후처리 범위에서 수리하되 등록된 모델·학습 설정을 바꾸어 게이트를 우회하지 않는다.
- 자동 확인: 이 대화의 `jena-p` heartbeat, 30분 간격. 사용자 조치가 필요 없는 반복 상태 알림은 생략하며, 최종 전달 후 삭제한다.

## 검증 완료 범위

- 새 분기 구현/초기화/RNG/gradient/선택 순서/범위 검사: CPU 테스트 15개 PASS.
- 독립 읽기 전용 코드 검토 PASS. 실제 CUDA 학습의 성공 여부는 실행 결과로 별도 확인한다.
- 실제 Jena pre-T 로더: 16,772행, 21채널, V origin 43개, 마지막 V 타깃 끝 16,739. T 시작 16,772 이전.
- 기존 OURS10 + 연장된 No-P10 + A2 10의 예측 파일 해시와 정규화된 타깃·cohort·FP64 MSE/MAE 검산 PASS.
- 기존 연구 Python 및 두 이전 캠페인 보고서 전체와 재사용 예측 파일 5,588개를 보존 해시 목록에 기록했다.
- 위 검증은 학습 또는 최종 발표 자료의 완료를 뜻하지 않는다.

## 결과와 발표

- controller는 학습 종료 후 `postprocess.py`를 실행한다. `SUMMARY.csv`, `CONTRASTS.csv`, `PAIRED_DIFFERENCES.csv`, `FIG_JENA_P_BRANCHES.*`, `RESULTS_KO.md`, `ANALYSIS_VALIDATION.json`이 생성된다.
- `TRAINING_COMPLETE`는 기본 20개 완료 상태다. 연장은 `extension_status`로 구분한다. 부분 연장은 기본 결과와 섞어 n을 늘리지 않는다.
- 발표 자료는 `presentation/`에 생성한다. 최신 편집 원본을 로컬/연결 소스에서 찾지 못했으므로 첨부 PDF의 19장 구성을 복원하고 수정 장은 네이티브 개체로 재작성한다. 이 한계를 README에 명시한다.
- 최종 이름: `Temporal_Channel_Input_Adaptation_revised.pptx`, `.pdf`, `SPEAKER_NOTES_KO.md`, `CHANGELOG_KO.md`, `README.md`.
- 원 첨부 PDF와 기존 결과는 보존한다. `TBD/연장 중` 등의 낡은 표현은 새 결과와 기존 완료 기록으로 정확히 갱신한다.
- 마지막 검수에서 실제 수치, 19장 PPTX/PDF, 문제정의·방법론과16·18장 렌더링, 발표 대본, 탐색/용량/학습 상한/기존 비교 잠정성 표기를 확인한다.
- 검수 후 `FINAL_REVIEW.json`에 실제 검토 파일과 해시를 기록한다. `package.py`가 ZIP CRC와 모든 manifest 해시를 검증한다.
- 목표 완료 처리는 기본 20개 결과 및 요청된 발표 수정의 실제 산출물을 확인한 뒤에만 한다.

## 2026-10-27 23:00 KST 검수 진행

- 첫 실제 시간 분기 실행은 exit 0, 약 696.93초에 완료했다. 다음 채널 분기 실행은 별도 worker로 순차 진행한다. PID는 고정값으로 추정하지 말고 PROGRESS/LIVE_WORKER와 현재 OS에서 다시 확인한다.
- 독립 감사로 후처리의 기본20 누락 검출을 보강했다. 기본 결과가 빠졌는데 연장 결과로 n10이 채워져 통과하는 경우를 차단한다. 집중 테스트19개 PASS(담당 worker 실행), root는 첫 실제 완료 job의 공개 증거9개와 V cohort 및 FP64 지표 재계산을 별도로 확인했다. POSTPROCESS_EVIDENCE_REVIEW.json 참고. 전체20 검증 완료를 의미하지 않는다.
- 후처리 그림은 실제 cap/연장/PROVISIONAL을 표시한다. 학습/선택/수치/판정 기준 및 봉인 과학 소스는 그대로다.
- 발표 초안은 PowerPoint 일반 열기, PDF19장, 내부노트19개, 전장 렌더 기술검증을 통과했다. 문제정의/방법론과 기존 비용/분위수/K 문구를 실제 출처에 맞게 정리했다. A_fit0–50%→A_stop50–60%선택→A60재학습→V70–80%평가이며, 9장 분산은 A_fit이다.
- 최종 발표 생성 명령: `& reports/jena_p_branch_20261027_v1/presentation/finalize.ps1 -Final`. ANALYSIS_VALIDATION PASS와 실제20/5종n10 전에는 최종을 생성하지 않는다. 스크립트는 native chart IQR, Office 정규화, 일반열기,19장렌더,실제chart XML/노트/PDF16·18 검사를 수행한다.
- 최종 root가 FINAL_REVIEW.json을 만들 때 required reviewed_files(repo-relative POSIX SHA map): PPTX/PDF/노트/CHANGELOG/README, PRESENTATION_VALIDATION.json, BUILD_FINAL.json, OFFICE_OPEN_FINAL.json, ANALYSIS_VALIDATION.json, EXECUTION_COMPLETION.json, SUMMARY.csv, CONTRASTS.csv, RESULTS_KO.md, FIG_JENA_P_BRANCHES.png. 반드시 실제 내용을 보고 기록한다.
- package.py는 위 최종검수 범위와 analysis.verified_files(기본/연장 성공job의 공개증거,분석입력,결과물) 모든 SHA를 다시 확인한다. PPT최종상태/19장·노트/출력해시/정상열기도 확인한다.
- ZIP에는 최종 산출물과 source assets20개, SOURCE_RECOVERY/BUILD_FINAL/CHANGE_MAP, build/validate/export/prepare/finalize 코드 및 OFFICE_OPEN_FINAL을 포함한다. 초안/probe/repaired/임시렌더는 제외한다. PACKAGING_REVIEW.json 참고.
- 아직 남은 작업: 기본20 및 조건부연장 종료→전체분석 검증→실제결과로 최종PPT/PDF 생성→수치·문구·실제렌더 검수→FINAL_REVIEW→package→독립CRC/manifest 재검사→최종전달 및 jena-p 자동화삭제/goal완료. 새 commit/push는 하지 않는다.

## 최종 산출물 검수 (2026-10-28)

[확인] 기본 20회가 모두 첫 시도에 완료됐다. 시간 분기 cap3 선택은 4/10, 채널 분기는 1/10으로 등록된 연장 기준에 해당하지 않아 추가 연장은 0회다. 새 20개와 기존 30개의 V 예측을 독립 FP64로 재계산해 공개 표와 대사했다. 보호 파일 5,588개와 등록 과학 소스 5개의 해시를 확인했다.

[확인] 19장 최종 PPTX/PDF, 한국어 대본과 변경 대조를 생성했다. 문제정의·방법론과 16·18장에 실제 결과를 반영했다. 전체 P 대비 시간 분기만 구성의 MSE 중앙값 차이는 +0.17%로 불명확하며, 채널 분기만 구성은 +8.92%다. 시간축의 본질적 우월성·두 분기의 필연성·동등성을 주장하지 않는다. 탐색 V, 용량 차이, 실제 cap10/cap3 차이, 기존 비교 방법의 잠정성을 표시했다.

[확인] 정상 PowerPoint 열기, 19장/19개 노트, 차트·IQR·문구 검증과 실제 시각 검수를 완료했다. 원본 PDF 기반 재구성 및 일부 이미지 보존의 한계는 presentation/README.md에 설명한다. FINAL_REVIEW.json은 이 검수에서 읽은 실제 산출물의 SHA256을 기록한다.

ZIP 검증의 최종 상태와 해시는 BUNDLE_RECEIPT.json 및 INDEPENDENT_BUNDLE_VALIDATION.json을 확인한다. 이전 절의 진행 중·남은 작업 설명은 당시 기록이다. 새 commit/push는 하지 않았다.
