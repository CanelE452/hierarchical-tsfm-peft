# 실행 코드·검산·게시 범위

이 묶음은 기존 실험 결과를 게시한다. 게시 과정에서 학습·원자료 역직렬화·F 추론·비용 재측정은 하지 않는다. 새 학습을 자동 실행하는 스크립트나 큐도 추가하지 않았다.

주요 실행 코드는 [projection_followup_v1](../projection_followup_v1/)에 있다. 원 계획·source snapshot·실패·선정 잠금·자원 영수증은 함께 보존한다. 원문 SHA는 작성 당시 파일을 가리키며 과거 source는 Git 이력과 해당 run의 `sources`에서 확인할 수 있다.

- [시간순 보정 실행](../projection_followup_v1/run_purged_readout_stability_v1.py), [독립 검산](../projection_followup_v1/audit_purged_readout_stability_v1.py).
- [STATIC/RAMP 실행](../projection_followup_v1/run_horizon_decoder_diagnostic_v1.py), [전체 비용](../projection_followup_v1/run_horizon_decoder_cost_v1.py), [수정된 비용 검산](../projection_followup_v1/audit_horizon_decoder_cost_v2.py).
- [오차항 진단](../projection_followup_v1/run_horizon_correction_error_terms_v1.py).
- [예측 cross-moment](../projection_followup_v1/run_predictive_cross_moment_v1.py), [native](../projection_followup_v1/run_new_direction_native_v1.py), [직접 ridge](../projection_followup_v1/run_new_direction_direct_ridge_v1.py), [규제 확장](../projection_followup_v1/run_ridge_regularization_extension_v1.py).
- [고정 Q](../projection_followup_v1/run_orthogonal_basis_controls_v1.py), [Q 학습](../projection_followup_v1/run_orthogonal_q_learning_v1.py), [분위 요약](../projection_followup_v1/run_quantile_point_summary_v1.py).

각 실행기는 현재 실제 입력·이전 PASS·source SHA에 결속하고 새 tag를 요구한다. 기존 tag를 덮거나, 비어 있는 gate를 임의로 채우거나, 다른 자료를 같은 이름으로 넣어 재실행하지 않는다. 보호 TEST·설치/다운로드·큰 백본 학습·기존 작업 종료는 이번 게시 요청에 포함하지 않는다.

## 큰 JSON 복원

[EVIDENCE_ARCHIVES.json](EVIDENCE_ARCHIVES.json)은 이전6개와 이번3개 JSON archive의 원 경로·원 SHA·압축 SHA·byte 수를 제공한다. `.gz`를 풀면 원 JSON과 byte가 같다. 원본은 기존 로컬 경로에 그대로 남아 있다. clone에서 필요한 경우 다음 stdlib 도구를 쓸 수 있다.

```powershell
python research/bolt_compatible_adaptive_20261014/publication_main_20261015_v2/restore_evidence.py
python research/bolt_compatible_adaptive_20261014/publication_main_20261015_v2/restore_evidence.py --restore
```

첫 명령은 archive와 존재하는 원본을 검사한다. 두 번째는 없는 JSON만 복원하며 기존 파일을 덮지 않는다. 과학 결과를 새로 생성하는 명령이 아니다.

<a id="local-assets"></a>

## 로컬 전용 자산

원자료·safe NPZ·예측/계수 NPZ·모델 binary·checkpoint는 로컬에 남긴다. 보고서의 로컬 자산 링크가 이 절로 연결되면 GitHub에서 누락된 성능 표라는 뜻이 아니라 비게시 재현 자산이라는 뜻이다. 실제 적합·F 추론을 재현하려면 계획의 정확한 로컬 자산과 해시가 모두 필요하다. 공개 코드와 합성 테스트만으로 전체 과학 실험이 재현됐다고 주장하지 않는다.

직접 인용한 과거 텍스트·코드 8파일은 [references 안내](references/README_KO.md)에 따로 복사했다. 해당 역사 문서의 추가 의존성 전체를 게시한 것은 아니며, 빠진 역사 텍스트와 비게시 binary 재현 자산을 구분해 안내한다.

게시 보고서는 원본 Markdown의 링크와 상단 안내만 변환했다. 과학 manifest의 원 MD SHA와 게시 사본 SHA는 다르며 [최종 REPORT_COPY_MAP_v2.json](REPORT_COPY_MAP_v2.json)에서 구분한다. 첫 초안 변환 기록도 보존한다. CSV·PNG·SVG·원 manifest 사본은 byte 동일하다. 게시 검증은 Git index/파일/형식/링크/압축 복원/민감 패턴 범위이며 과학 검산과 별개다. 게시 직전 관련 합성·진입 회귀 검사는36 tests·23 subtests를 통과했다. 이 검사에서 실제 과학 학습·F 추론을 다시 실행하지 않았다.

최종 게시 범위는 [FILES_v2.json](FILES_v2.json)이다. 초기 FILES와 자기 영수증 생성 전 링크1개 실패, 이후 PASS를 보존했다. 게시 재검사는 새 출력 경로를 지정하며, CSV 두 파일이 각각18.1MiB여서 파일 상한을32MiB로 명시한다. archive9개의 원 JSON이 없는 clone은 먼저 위 도구의 `--restore`로 byte 동일 JSON을 복원해야 한다.

```powershell
python research/bolt_compatible_adaptive_20261014/publication_main_20261015_v2/verify_publication.py --manifest research/bolt_compatible_adaptive_20261014/publication_main_20261015_v2/FILES_v2.json --output .cache/fresh_publication_review.json --max-file-mib 32
```

이 명령은 과학 학습·모델 추론을 수행하지 않는다. 생성된 최종 영수증의 GitHub 경로는 [색인](README_KO.md)에 연결한다.
