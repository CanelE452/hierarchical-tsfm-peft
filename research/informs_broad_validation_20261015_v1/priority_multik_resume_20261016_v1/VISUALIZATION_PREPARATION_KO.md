# Multi-K 그림 생성 코드 준비

현재는 **PREP_ONLY**다. 이 문서는 `plot_multik_resume.py`, `test_plot_multik_resume.py`와 함께 준비한 코드의 범위를 기록한다. 실제 N7 결과 수치·감사/보고 입력·통계/trial·모델/Task/NPZ/CP를 읽지 않았고 실제 SVG/PNG/그림 결속은 생성하지 않았다. 진행 중 학습과 프로세스를 조회하거나 변경하지 않는다. 기존 source48·PLAN·결과·타인의 파일은 보존한다.

## 입력과 실제 실행 조건

수치 입력은 ROOT가 먼저 완료한 `MULTIK_RESUME_RESULTS_v1.csv`와 `MULTIK_RESUME_BINDINGS_v1.json` 두 파일뿐이다. 원 감사 JSON·공개 충분통계·비용 trial이나 재사용 원 CSV를 renderer가 다시 해석하지 않는다. 보고 bindings의 실제 감사 SHA·정확한 producer SHA·완료 상태·새 부모2/출력6/비용12·재사용 PCA3/independent K6·전체 MASTER 미완료와, 그 bindings에 적힌 CSV SHA를 확인한다. 코드/helper와 두 입력의 전후 SHA도 대사한다.

현재 builder는 ROOT 요청에 따라 문구만 정확히 한 `d616ce65839393e0fdb70bb3a98e557d29870ef62428204436a66429f5516f37`, auditor는 `0c22445d86e75610d78caeda0dc645c0fa7c09d3ed46f6a2b9ed44878ea25146`다. 변경 전 builder `3617ff09…`는 private exact-byte 백업으로 보존했다. 원 input helper `8bb730a5…`와 style의 ResourceCheck `50e25501…`만 재사용하며 기존 helper를 수정하지 않는다.

미래 실제 `report_bindings_SHA`, `CSV_SHA`, `actual_audit_SHA`는 **null/미확정**이다. 기본 실행은 입력/출력0·NumPy/Torch/Matplotlib import0으로 반환한다. 실제 실행은 ROOT가 과학 작업 종료와 quiet window를 확인한 뒤 정확한 SHA를 모두 명시해야 한다.

```text
plot_multik_resume.py --execute --root-triggered
  --public-root <R> --cache-root <C>
  --plot-sha <reviewed current renderer source SHA>
  --report-binding-sha <actual report bindings SHA>
  --csv-sha <actual original-precision CSV SHA>
  --audit-sha <actual audit SHA already bound by that report>
```

원 마감1792222550·자체 RSS1GiB·CPU/BLAS1을 기존 ResourceCheck로 유지한다. 새 설치·마감/예산 확대는 없다. Matplotlib과 기존 Malgun font는 실제 renderer 안에서만 불러오며 현재 준비에서 import/render하지 않는다.

## 표시 계약

2×2 패널은 normalized channel-macro MSE·MAE, CPU2 B1/B8 ordinary whole20 median의 **초/batch**다. 같은 Beijing C11/L512/H48·seed92601·반복 개발 V43/20539 관측 라벨이다. shared/max-only/PCA는 실제 K3/6/11의 세 선이고, independent는 기존 K6의 한 점만 표시한다. 세 새 독립 모델로 잘못 세지 않으며 independent K3/K11을 생성하거나 추정하지 않는다. 모든 경로는 Bolt를 포함한다.

새 shared/max 두 부모에서 나온6개 출력과, 재사용 fixed nested PCA3/independent K6 한 점을 색·marker·linestyle·라벨 및 artist `row_scope/run_ids`로 구분한다. 색각 접근성을 고려한 색과 서로 다른 표식을 함께 쓴다. y축은0부터이고 선은 관측 K를 연결하는 시각적 표시일 뿐 중간 K 성능을 측정한 것이 아니다. 16개 실제 artist의 K 위치와40개 y값이 CSV float와 정확히 같은지 확인하고 원 CSV 문자열도 결속에 보존한다. 표시용 반올림을 원 값에 적용하지 않는다.

PCA는 다른 D이고 shared/max의 최대 K11 구조·PCA 초기화가 같아도 K별 학습 노출/F 연산·총 compute는 다르다. 원 independent K11 identity 초기화와의 confound, independent K3/K11 미실행, 제한 학습·반복 V·단일 seed·수렴/독립 일반화 미확정을 명시한다. CI·seed 분산·유의성·p90 band·전환/저장량/메모리 효과는 추가하지 않는다. warm whole에서 기존 F pool cold load/Route loading을 제외하며 B8을8로 나누지 않는다. F-only resident와 native receipt 미기록 raw F/proofmap/dtype는 NA로 남긴다. 전체 E4/56/MASTER 미완료다.

## 향후 CreateNew 출력

- 공개 `figures/MULTIK_RESUME_v1.svg`
- 공개 `MULTIK_RESUME_VISUALIZATION_KO_v1.md`
- 공개 `MULTIK_RESUME_FIGURE_BINDINGS_v1.json`
- private cache-relative `priority_multik_resume_20261016_v1/visualization/MULTIK_RESUME_v1.png`

기존 출력이 하나라도 있으면 입력을 읽기 전에 거부한다. 보고서를 먼저 완성하고 그림은 그 보고를 설명한다. renderer 성공을 과학적 성공·우위·수렴으로 바꾸지 않으며 그림 실패는 독립 과학 경로의 중단 gate가 아니다.

## 준비 검증

구현 부재 RED1/skip8 후 GREEN을 확인했다. 최종 stdlib 합성10개 PASS: 기본0 I/O·과학 라이브러리 import0, 정확10행/3선+1점, 16 artist/40 원정밀도 값, missing/duplicate/K/scope·partial/잘못된 SHA·추정 native/메모리값 거부, 기존 출력 보존을 검사한다. 현재 builder의 실제 `compose` API도 작은 가짜 감사/참고로 연결해 CSV schema 호환을 확인했다. 실제 보고·40개 실제 값·Matplotlib/font/ResourceCheck 실행·레이아웃과 이미지 검토·CreateNew 파일 integration은 **아직 미실행/미검증**이다.
