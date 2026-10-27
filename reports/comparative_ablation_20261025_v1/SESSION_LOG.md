# 실행 준비 세션 기록

이 문서는 2026-10-25 16:23 KST 무렵의 준비 단계 스냅샷이다. 이후 실제 실행 현황은 controller의 실행 시계·로그·결과 receipt를 따른다. 이 스냅샷에서 **새 실제 학습은 시작하지 않았다**.

## 작업 위치와 승인

[확인] 최초 작업 폴더 `E:/CODING/proj/hierarchical-tsfm-peft`는 main의 `c7a06e522e1cd8e100a3e2291a7f693dd3f7125f`였고, 입력 계획이 요구한 Kτ 보고서가 없었다. `git worktree list`로 `adapter-diagnostics-v2` worktree의 `769db577da01bb2a7ccbd179909d7e12fd13c7fe`와 필요한 자료를 찾았다. 해당 checkout은 다른 chat 소유여서 기존 내용을 변경하지 않고 같은 commit에서 별도 `comparative-ablation-20261025` worktree를 만들었다. 근거: `GIT_PREFLIGHT.json`.

[확인] 사용자의 직접 답변 **“로컬 커밋 후 실험 진행”** 후 사전등록 MD·JSON·RUN_PLAN 세 파일만 로컬 commit했다. commit은 `b92f14e4ed18291aa0726eb660f41e3bd9903dde`, git commit 시각은 2026-10-25 16:14:27 KST다. `PREREGISTRATION_COMMIT.json`의 `committed_utc`는 후속 receipt 기록 시각이며 git commit 시각과 구분한다. push는 하지 않았다.

## 확인된 준비 결과

[확인] OURS 40개와 Raw 4개의 기존 V 배열·관련 receipt를 찾고 FP64로 재검산했다. 최대 MSE 차이는 `2.220446049250313e-16`, MAE 차이는 `1.6653345369377348e-16`이다. Kτ 구성은 34개 신규와 6개 역사적 재사용이며 Appliances K10은 10개 모두 신규다. 역사적 seed 92601/92602의 선택 epoch는 Jena 1/1, ETTh1 .25/.25, ETTh2 1/3이다(`G1_REUSE_AUDIT.json`).

[확인] 현재 F SHA, 4개 데이터의 pre-T 지문·분할·A_fit/A60 정규화가 일치했다(`G1_RUNTIME_AUDIT_ATTEMPT003.json`). 최초 두 감사 시도는 Windows NUL/tempfile 쓰기 차단을 과도하게 적용한 감사 도구 오류로 데이터 읽기 전에 끝났다. 두 기록은 보존했고, 성공 시도의 임시 쓰기는 새 캠페인 `.cache`에 제한했다.

[확인] Jena Raw의 이전 `UNRESOLVED_HISTORICAL_F_STATE`는 원 manifest의 복합 지문을 후속 verifier가 직접 F 지문으로 풀지 않았던 메타데이터 공백이었다. 원 생성 코드와 F/scaler/data 복합 hash를 검산해 해결했다. 재추론은 필요하지 않았다(`G1_JENA_RAW_PROVENANCE.json/.md`).

[확인] C1/C2/A1 × 4개 데이터 순전파는 12/12 통과했다. PCA의 최대 재구성 차이는 `3.7643758243888215e-7`로 `1e-4` 기준 이내다. 이 실행은 V 한 배치의 shape/finite 검사만 했고 V 점수·모델 선택·학습은 하지 않았다.

[확인] G5는 기본 284개와 잠재 연장 280개, 합계 564개 spec을 확인했다. 1,980개 필드 비교를 공개하며 core6 파일은 유지됐다. C2/C3의 실제 초기화는 PyTorch 기본값이므로 init_kind 메타데이터를 PYTORCH_DEFAULT로 명시했다. 모델 동작을 바꾼 수정은 아니다(`G5_REVIEW.json`).

[추정] 새 예산 계산은 비교 C1–C3 상한 22.196984408893492시간, 전체 기본 기대 38.22656998951849시간, 모두 상한 checkpoint를 선택한 기본 상한 48.60988158667703시간이다. 최악의 전 그룹 연장은 추가 119.48240905187869시간이므로 모두 수행할 수 있다고 약속하지 않는다. 원 계획의 30/35시간 추정은 낙관적이었다. 비교가 40시간 이내라는 사전 게이트는 통과하며, 실측 후 낮은 우선순위 family를 제외하는 규칙이 필요하다(`G6_BUDGET.json`).

[확인] root는 16:20 KST에 기존 44개 실행의 A0·오차분포 계산을 종료코드 0으로 완료했다고 보고했다. 원 단위의 정확한 0을 처리하는 MAPE 보정과 FP32 target 왕복 검산이 포함됐다. 산출물은 `METRICS_EXISTING.csv`, `ERROR_DISTRIBUTION_EXISTING*.csv`, `VMSE_DEFINITION.md`, `OPENED_FILES_EXISTING_METRICS.json`이다. 분석 모듈의 후속 시험은 17개 통과로 보고됐다.

## 검증과 남은 단계

[확인] 보고된 시험 배치 26개(root 초기), 13개(models), 13개(runner), 15개(analysis 이전), 11개(controller)는 서로 겹치거나 코드 시점이 다르므로 합산하지 않는다. 분석 후속 17개 결과는 이전 15개를 대체하는 시점의 결과다. 최신 통합 실행에서는 controller fixture가 새 완료파일 검사에 맞지 않는 문제가 나와 fixture를 보완 중이다. 최신 통합 PASS 개수는 아직 선언하지 않는다. 최종 확정은 root의 새 실행 결과로 기록한다.

[확인] Python 파일-open 이벤트와 제한된 numeric prefix loader의 근거를 남겼다. 모든 OS/native I/O를 감시했다고 주장하지 않는다. Jena 압축 원본 전체의 불투명 바이트 hash는 읽지만 보호 T 값은 숫자로 파싱하지 않는다. 실제 학습 전 최신 통합 검증·생성 코드 결속·기존 파일 hash 보존을 다시 확인해야 한다.

기존 `_docs/history`에는 새 내용을 쓰지 않았다. 이번 기록과 새 코드·보고서는 요청한 캠페인 이름공간에만 둔다.
