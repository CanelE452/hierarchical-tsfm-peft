# 게시 보조 자료 안전 공개본

원 log/diff8개를 전체 텍스트 .txt view로 보존했다. 원 byte SHA·공개 SHA·치환 라인은 [원본 결속](VIEW_PROVENANCE.json)과 [alias9행](ALIAS_ROWS.json)에 있다. 실제 기계 prefix가 있으면 해당 prefix만 치환하는 규칙이며, 이번8개는 이미 안전한 텍스트로 byte-identical이다. 숫자·결과·PASS/FAIL·오류 traceback·patch 행은 생략하지 않았다. 원본은 로컬 보존하고 공개 gzip으로 우회하지 않았다.

[원 비용 auditor의 정확한 복사본](audit_core_costs.py)과 [안전 합성 fixture 테스트 복사본](test_audit_core_costs_public.py)은 이 새 폴더에만 둔다. 원 테스트의 static absolute fixture는 실제 사용자 경로가 아닌 합성 입력이지만 공개 경로 검사와 충돌하므로 무관한 합성 드라이브를 동적으로 만든다. AST상 해당 assignment/원소 외 일치하고 동일7개 테스트를 [실제 실행](TEST_RESULT.json)해 통과했다. 기존7개 검사와 합산하거나 실제 비용 재측정으로 부르지 않는다.

```text
python research/informs_broad_validation_20261015_v1/diagnostics/publication_safe_aux_views_20261016_v1/test_audit_core_costs_public.py
```

[소스 변경 한계](SOURCE_PATCH_PROOF.json), [7개 실제 stdout/stderr](SAFE_TEST_7_PASS.txt), [링크 조사](MISSING_LINK_REVIEW.json)를 확인한다. 원본MD·source39·registry·lock·결과·후보·Git은 바꾸지 않았다. 새 테스트 복사본 허용 예외일 뿐 원 과학 소스를 고친 것이 아니다. 명령 README 링크는 결과-only README로 바꾸지 않고 canonical 재현 절 연결안을 ROOT에 제시했다. 구형 그림 링크는 ROOT가 처리한다.
