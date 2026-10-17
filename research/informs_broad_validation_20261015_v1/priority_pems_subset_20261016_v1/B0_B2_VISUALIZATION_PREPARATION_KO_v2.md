# PeMS08 C32 B0/B2 그림 준비 — 실제 렌더 0

이 파일은 source-only PREP다. 실제 B0/B2 학습·추론·검산·비용·감사·보고·그림 완료를 뜻하지 않는다. 미래 audit/report/CSV SHA는 아직 미정이며 ROOT가 실제 종료·검산·보고 후 제공한다. 기존 source·결과·계획은 변경하지 않았다.

`plot_b0_b2_v2.py`는 frozen [auditor](audit_b0_b2.py) `c002a243431c6b3a8304dcd074c26ebec8b05e0115a8c95e3763286e210c9d66`와 [builder](build_b0_b2_report.py) `e6866977ffef406cf18c68aa65d7717d9f56072372224d04277ce4497e95f89d`에 결속한다. 기존 Raw/G plot의 표시 원칙을 적용하되 새 B0/B2 schema를 사용한다. 실제 `PASS` 또는 유효 peer가 남은 `PARTIAL_VALID_EVIDENCE_PRESERVED`만 받는다. 전역 거부 감사는 그림 입력으로 승격하지 않는다.

MSE·MAE·CPU2 warm B1/B8 ordinary whole20 median의 4 panel이다. C32/L512/H12/V148/seed92601·실제 관측 cohort의 Raw/G 재사용과 새 B0/B2 K8 독립 학습을 구분한다. Raw/G가 없거나 peer·batch 비용이 미완료이면 해당 자리는 NA이고 bar를 만들지 않는다. 성공 시 최대16개 실제 artist 값을 canonical CSV와 exact 대사한다. 실제 선택 refit0도 유효하다. B8은 전체8개 입력 batch의 초이며8로나누지 않는다.

ROOT 실행 형식은 다음과 같다. 아래 placeholder는 실제 SHA가 아니다.

```text
python -B plot_b0_b2_v2.py --execute --root-triggered
  --public-root <R> --cache-root <C> --plot-sha <current-source-SHA>
  --report-binding-sha <actual-B0_B2_BINDINGS-SHA>
  --audit-relative priority_pems_subset_20261016_v1/<actual-audit>.json
  --audit-sha <actual-audit-SHA> --csv-sha <actual-canonical-CSV-SHA>
```

actual 실행에는 CPU/BLAS1 환경(OMP/MKL/OPENBLAS/NUMEXPR=1), 원 외부 CPU2/RSS3GiB guard, 원 deadline `1792222550`과 저장 상한이 적용된다. pinned `plot_beijing.py` ResourceCheck는 자체 RSS1GiB·동일 deadline을 검사한다. 이미 설치된 Malgun 글꼴만 사용한다. 기본 호출은 I/O0, renderer import0이다.

canonical CSV와 한국어 MD를 frozen `builder.compose`로 바이트 재구성해 실제 보고 결속과 비교하고, audit·report·reference의 모든 captured input SHA를 전후 검사한다. 실제 통계/trial이나 원자료를 다시 계산하지 않는다. 원 audit/report/reference 입력을 opaque 바이트로 열어 SHA를 대사하는 I/O는 있다. 원자료·T 수치/배열·NPZ/CP 텐서를 해독하거나 모델을 로드하지 않는다. 원 p90·CI를 새로 추가하지 않는다.

새 출력은 CreateNew만 허용한다.

- 공개 `figures/B0_B2_FOUR_PANEL_v1.svg`
- 공개 `B0_B2_VISUALIZATION_KO_v1.md`
- 공개 `B0_B2_FIGURE_BINDINGS_v1.json` (`PEMS_B0_B2_FIGURE_BINDINGS_v1`, 출력 경로는 namespace-relative)
- private cache `priority_pems_subset_20261016_v1/visualization/B0_B2_FOUR_PANEL_v1.png`

그림 결속에는 source/audit/report/CSV SHA와 전체 captured input 목록, 실제 artist·missing 상태, private preview SHA·범위, 원 자원 범위를 남긴다. private 절대경로나 원 데이터 값은 공개하지 않는다. 실패 시 기존/부분 파일을 덮지 않으며 보고가 먼저다. 그림 실패는 다른 유효 peer나 독립 경로를 막는 gate가 아니다.

검사 수준은 stdlib 가짜 audit/CSV·mock axis에 한정한다. source 부재 RED11 → GREEN11 (0.016초), 기본 I/O0와 NumPy/Torch/Matplotlib import0을 확인했다. 검사 항목은 exact producer/role/cohort/F폭, 재사용 scope, 실제 선택0, partial/참조 부재·missing 비용, ordinary20 median, artist 값·누락, 기존 출력 보존이다. 실제 rendering·폰트·레이아웃·실제 future 입력 closure 통합은 미검증이다.

한 seed·반복 개발 V·제한 search·각기 다른 P/용량/선택 refit 노출의 대조다. 구조 인과 효과·동등 compute·수렴·독립 일반화·Bolt 필요성을 확정하지 않는다. Raw/G는 새 독립 표본이 아니다. F-only resident와 parameter numel은 미기록 NA이며 전체 Route memory로 대입하지 않는다. upstream 미래 보간·전처리 계보는 UNKNOWN이고 전체 subset·MASTER는 미완료다. 실제 모델·자료·재채점·비용·렌더·프로세스·Git·network 작업은0이다.

## v2 표기 정정과 테스트 게시 범위

원 dd0c 그림 source·f342 테스트·eed6 준비 문서 및 7957 감사 테스트는 보존한다. 새 그림 v2는 opaque-byte hash I/O 범위와 자기 source filename/code URL만 정정했다. scientific/numeric/render/resource gate·schema·v1 출력명은 그대로다. 실제 렌더·실험 수행은0이다.

감사 테스트는 새 `test_audit_b0_b2_v2.py`만 게시 후보로 사용한다. 원 가상 절대경로 literal을 `TemporaryDirectory` fixture로 바꾸었고 원13개 감사 검사의 논리와 결과는 같다. 원 `test_audit_b0_b2.py`와 그것을 연결한 원 `B0_B2_PREPARATION_KO.md`는 로컬 보존하며 게시 선택에서 제외한다. 이는 표기·게시 호환 정정이며 실험·성능·학습 오류를 수정한 것이 아니다.
