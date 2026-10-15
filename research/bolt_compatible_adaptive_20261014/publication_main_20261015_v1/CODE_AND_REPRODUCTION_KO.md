# 코드·근거·재현 조건

이 게시 작업은 이미 저장된 결과를 정리한 것이다. 게시를 위해 학습·F 추론·비용 타이머를 새로 실행하지 않았다. 공개 MD/CSV/PNG/SVG 및 JSON·로그·SOURCE와 독립 검산 기록을 함께 읽는다. 개별 상태·마스크·seed·cohort가 다른 결과의 절대 시간을 합쳐 모델 순위를 만들지 않는다.

## 실행 코드에서 결과까지

실행 파일은 `projection_followup_v1` 원경로에 보존했다. 파일마다 실제 입력과 source snapshot SHA, 고정 값, 새 출력 tag, 오류 보존과 TEST guard가 있다. 과거 실행 파일을 편집해 새 결과를 만들지 않았다.

- PCA/ForeCA: [수학](../projection_followup_v1/projection_math_v1.py), [수선 후 실행](../projection_followup_v1/run_projection_followup_v2.py), [native head](../projection_followup_v1/native_head_v1.py), [보고서](reports/matched_projection_v1/REPORT_KO.md)
- 고정 D와 rank-K 진단: [fixed-D span](../projection_followup_v1/fixed_decoder_span_v1.py), [미래를 아는 rank-K 진단](../projection_followup_v1/oracle_rank_ceiling_v1.py), [보고서](reports/oracle_rank_addendum_v2/REPORT_KO.md)
- D-only pinv(E): [실행](../projection_followup_v1/decoder_metric_probe_v1.py), [전체 비용](../projection_followup_v1/run_decoder_metric_cost_v1.py), [보고서](reports/decoder_metric_intervention_v2/REPORT_KO.md)
- FIT target 공간: [표현 수학](../projection_followup_v1/future_target_projection_v1.py), [실행](../projection_followup_v1/run_fit_target_comparison_v1.py), [보고서](reports/FIT_target_comparison_v1/REPORT_KO.md)
- 과거 constant gate: [gate](../projection_followup_v1/past_constant_gate_v1.py), [실행](../projection_followup_v1/run_past_constant_gate_v1.py), [전체 비용](../projection_followup_v1/run_past_constant_gate_cost_v1.py), [보고서](reports/past_constant_gate_v1/REPORT_KO.md)
- local D와 shrinkage: [local D](../projection_followup_v1/local_past_decoder_v1.py), [실행](../projection_followup_v1/run_local_past_decoder_probe_v1.py), [shrinkage 실행](../projection_followup_v1/run_local_decoder_shrinkage_v1.py), [새 matched head](../projection_followup_v1/run_matched_shrunk_native_v1.py), [통합 보고서](reports/shrinkage_matched_native_v1/REPORT_KO.md)
- 원 채널 subset: [수치 수선 후 선택 수학](../projection_followup_v1/raw_subset_projection_v2.py), [실행 v2](../projection_followup_v1/run_raw_subset_comparison_v2.py), [보고서](reports/raw_subset_comparison_v1/REPORT_KO.md)
- lag/목적 2×2: [과거 prefix·lag 수학](../projection_followup_v1/causal_lag_decoder_v1.py), [실행](../projection_followup_v1/run_causal_lag_objective_v1.py), [전체 비용](../projection_followup_v1/run_causal_lag_cost_v1.py), [보고서](reports/causal_lag_objective_v1/REPORT_KO.md)
- 선형 E 갱신: [Torch adapter·손실](../projection_followup_v1/linear_e_adaptation_v1.py), [probe/학습 실행](../projection_followup_v1/run_linear_e_adaptation_v1.py), [전체 비용](../projection_followup_v1/run_linear_e_cost_v1.py), [보고서](reports/linear_e_adaptation_v2/REPORT_KO.md)
- F-input gradient: [stop-gradient 실행](../projection_followup_v1/run_f_input_gradient_ablation_v1.py), [실제 전체 비용](../projection_followup_v1/run_f_input_gradient_cost_v1.py), [보고서](reports/f_input_gradient_ablation_v1/REPORT_KO.md)
- 학습 뒤 공간: [진단 실행](../projection_followup_v1/run_learned_output_space_diagnostic_v1.py), [독립 검산](../projection_followup_v1/audit_learned_output_space_v1.py), [보고서](reports/learned_output_space_v2/REPORT_KO.md)
- RECON 초기화와 고정 D: [초기화 실행](../projection_followup_v1/run_recon_init_sensitivity_v1.py), [E-only 실행](../projection_followup_v1/run_fixed_recon_d_e_adaptation_v1.py), [E-only 보고서](reports/fixed_recon_d_e_adaptation_v1/REPORT_KO.md)
- 같은 L7 E/D native: [head](../projection_followup_v1/lagged_native_head_v1.py), [실행](../projection_followup_v1/run_matched_l7_native_v1.py), [전체 비용](../projection_followup_v1/run_matched_l7_native_cost_v1.py), [독립 과학 검산](../projection_followup_v1/audit_matched_l7_native_v1.py), [완료 보고서](reports/matched_l7_native_v2/REPORT_KO.md)
- 최신 잠재값 보정: [작은 직접 solver](../projection_followup_v1/fixed_l7_latent_calibration_v1.py), [과학 실행](../projection_followup_v1/run_fixed_l7_latent_calibration_v1.py), [독립 과학 검산](../projection_followup_v1/audit_fixed_l7_latent_calibration_v1.py), [실제 현재 상태](reports/fixed_l7_latent_calibration_snapshot_v1/REPORT_KO.md). 전체 비용·정식 보고 생성기는 SOURCE 준비 단계이며 실행 완료 자료로 세지 않는다.

개별 보고서의 하단 링크는 그 실행의 producer·원배열 감사·비용 감사·보고 감사와 source/plan/registry를 연결한다. 저장된 array/점수/로그와 lock을 감사한 범위, 실제 F·optimizer·타이머를 독립 재실행하지 않은 한계를 각각 적었다. 검산 건수가 독립 자료 수나 일반화 횟수는 아니다.

## 합성 검사와 실제 실행을 구분

아래 명령은 기존 실험을 다시 학습하는 명령이 아니라 작은 합성 fixture 검사다. 저장소 루트에서 기존 Python 환경과 의존성이 준비돼 있어야 한다. 이 게시 작업은 설치를 실행하지 않았다.

```powershell
.venv/Scripts/python.exe -m pytest research/bolt_compatible_adaptive_20261014/projection_followup_v1/test_fixed_l7_latent_calibration_v1.py -q -p no:cacheprovider
.venv/Scripts/python.exe -m pytest research/bolt_compatible_adaptive_20261014/projection_followup_v1/test_lagged_native_head_v1.py -q -p no:cacheprovider
```

실제 실행은 원 로컬 자산·정확한 SHA·등록된 부모 결과/감사·모델 runtime이 있어야 한다. 각 runner의 `--help`와 사전 `PLAN_*.md`, 실제 producer에 기록된 실행 계보를 먼저 확인해야 하며, 새 시도에는 기존 결과를 덮지 않는 새 tag가 필요하다. SOURCE의 exact-SHA gate를 꺼서 다른 입력으로 실행하면 원 실험 재현이라고 부를 수 없다. 학습 재시작·기존 프로세스 종료·보호 TEST 개방을 게시 권한으로 간주하지 않는다.

현재 계속 실행의 [승인·상한 해제 기록](../projection_followup_v1/AUTHORIZATION_SCOPE_v1.json), [continuation manifest](../projection_followup_v1/CONTINUATION_MANIFEST_v1.json), [현황 v9](../projection_followup_v1/CONTINUATION_STATUS_KO_v9.md)를 보존했다. v9 이후 최신 보정은 과학 적합과 독립 감사만 추가 완료했다. 과거 문서의 종료 시계와 미실행 표시를 현재 실행으로 바꾸지 않았다.

<a id="local-artifacts"></a>
## 공개하지 않은 로컬 자산

원자료·정규화 입력·물리 bridge·latent cache·선정 모델/optimizer checkpoint·`.npz`/`.pt`/`.pth`/가중치는 게시하지 않는다. 원 기록의 `path`, `sha256`, `arrays`의 shape/dtype/member SHA와 origin/mask/scaler 계보는 그대로 보존한다. 이 값은 재현에 필요한 로컬 자산의 식별자이지 GitHub에서 받을 수 있다는 뜻이 아니다. 그 자산이 없으면 clone만으로 실제 학습·추론·원배열 감사를 재현할 수 없다.

읽기용 Markdown에서 비공개 자산을 직접 가리키는 이동 링크가 있다면 이 절로 연결한다. 원 링크 target과 원보고서 SHA는 [사본 맵](REPORT_COPY_MAP.json)에 남긴다. CSV 내부 artifact 경로와 JSON provenance의 Windows 절대 경로는 원문 그대로이며, 공개 웹 이동 링크로 오인하지 않는다.

10 MiB를 넘는 미추적 대형 JSON은 원파일을 로컬에 보존하고 게시용 `evidence/`의 `.json.gz` 사본으로 연결한다. gzip은 내용의 손실 압축이나 요약이 아니다. 일반 JSON과 SOURCE·MD·로그는 원경로에 보존한다. 다운로드 후 gzip을 해제하면 원 JSON byte SHA를 대조할 수 있지만, 이를 위해 원데이터나 모델을 내려받을 필요는 없다.

## 원본과 읽기용 사본

[REPORT_COPY_MAP.json](REPORT_COPY_MAP.json)은 원본과 사본의 파일 경로·SHA·bytes·byte-identical 여부와 Markdown 링크 치환을 기록한다. 원 CSV/PNG/SVG/manifest는 byte 그대로 복사했다. MD는 숫자·본문을 유지하고 이동 링크만 바꿨다. 따라서 사본 MD의 SHA는 원 실험 manifest에 적힌 원 MD SHA와 다르다. 원 manifest는 원 실험의 보존 기록이고, 사본의 검증에는 사본 맵과 이번 게시 파일 manifest를 써야 한다.

스타일 보완 전 보고서, 중단된 native v1 그림 생성, subset Gram cancellation·수선 전 실행, 감사의 metadata/변수/형식 오류도 지우지 않았다. 오류를 모델 성능 실패나 새로운 독립 학습 조건으로 세지 않았다. 원본 SOURCE/결과를 고친 것과 새 버전으로 수선한 것을 각 보고서에서 구분한다.

[현재 결과 입구](README_KO.md) · [이전 게시의 코드·재현 설명](../publication_main_20261014_v1/CODE_AND_REPRODUCTION_KO.md)
