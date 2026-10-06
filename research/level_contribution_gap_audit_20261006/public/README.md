# 공개 근거 안내

이 폴더는 완료된 로컬 LEVEL 감사를 GitHub에서 읽을 수 있도록 만든 사본입니다. [전체 요약과 그림](../README.md), [최종 비판 보고](CRITICAL_REVIEW_KO.md), [증거 색인](evidence_index.csv)을 먼저 읽으면 됩니다.

- [초기 v1~v7 artifact 감사](early_artifact_checks.json)
- [후속 v8~v16 및 Peacock RAW artifact 감사](later_artifact_checks.json)
- [현재 LEVEL 계약의 Chronos-2 artifact 감사](chronos_artifact_checks.json)
- [원래 감사의 검증 receipt](verification.json)
- [공개 파일 목록과 양쪽 해시](publication_manifest.json)
- [실제 게시 사본 검증](publication_validation.json)

`verification.json`과 기존 `final_checks`는 **원래 감사·실행 시점**의 상태입니다. 그 안의 파일 크기·SHA·Git SHA·commit/push 미실시 문장을 현재 공개 사본의 검사나 현재 게시 상태로 해석하지 않습니다. 이번 게시 검증은 `publication_validation.json`에 따로 기록합니다. 자체 검사 통과는 독립 재현이나 논문 신규성을 인증하지 않습니다.

## 출처 찾기

`evidence_index.csv`의 출처는 원래 repository-relative 경로를 유지합니다. v1~v16의 기존 tracked 자료는 [감사 시작 시점 커밋](https://github.com/CanelE452/hierarchical-tsfm-peft/tree/2154e7f2b0e9878164e77536b8deaa9f372ac760/research)에서 읽을 수 있습니다. 당시 미게시였던 근거는 아래 사본 디렉터리에 원래 경로 구조로 담았습니다.

- [Chronos-2/Small 근거](source_snapshots/research/level_chronos2_controls_v1/): 정보 계약·revision·preflight·봉인·예측 manifest·평가·사용량 원장·같은 회차 비용·실제 완료 기록.
- [Peacock matched PCA RAW 근거](source_snapshots/research/tsfm_peft_peacock_matched_raw_v17_20261002/): protocol·부모 선택·두 seed의 실제 적합·평가·예측 manifest·검산.
- [전체 사본별 경로 대응](SOURCES.md).

예를 들어 색인의 `research/level_chronos2_controls_v1/accuracy_comparison.csv`는 공개본의 `source_snapshots/research/level_chronos2_controls_v1/accuracy_comparison.csv`에 대응합니다. `.cache/` 예측·checkpoint 경로는 로컬 보존 증거이며 공개 GitHub 파일이 아닙니다. 각 사본에 기록된 원본 SHA를 정리된 사본 바이트에 그대로 적용하면 안 됩니다. `publication_manifest.json`의 `original_sha256`과 `published_sha256`을 구분하십시오.

비용 aggregate의 CPU36·CUDA279개 dictionary는 기존 개별315개 JSON과 모두 동일함을 확인했습니다. 중복 개별 파일 대신 [CPU 원행](source_snapshots/research/level_chronos2_controls_v1/cost_cpu_rows.json)·[CUDA 원행](source_snapshots/research/level_chronos2_controls_v1/cost_cuda_rows.json), grid·binding·summary·CSV를 포함합니다. 큰 JSON이 GitHub 화면에서 접히면 Raw 보기로 읽을 수 있습니다.

## 그림 재생성

`../figures/build_figures.py`는 로컬 기존 CSV가 있으면 그것을 사용하고, 없으면 공개 사본 CSV를 사용합니다. 저장된 행 선택과 표시 변환만 수행합니다. 기존 `figure_values.json`은 최초 그림 생성 때 읽은 **원본 CSV**의 SHA와 행을 기록합니다. 비용 CSV의 개인 경로를 정리했으므로 공개 사본을 읽어 새로 렌더하면 CSV SHA는 달라질 수 있지만 표시 수치는 같습니다. 공개 사본은 전체 실험의 실행 가능한 배포 패키지가 아니라 선별된 정적 근거입니다.
