# 공개 충분통계 숫자 토큰의 무손실 인코딩

원 CSV의 모든 행·순서·token·header·개행과 원 gzip bytes를 보존했다. 부동소수점은 원문이 `str(float(token))`과 EXACT 같을 때만 IEEE754 binary64를 쓰며 float32·반올림을 사용하지 않는다. 정수는 canonical signed int64만 사용하고 다른 원문은 literal UTF8로 남겼다. 복잡한 quoted CSV 또는 재현 불가능한 gzip은 원 gzip 그대로 저장한다. 원 CSV/gzip SHA가 다르면 복원 PASS가 아니다.

- [새 FORMAT manifest](MANIFEST.json)
- [전수·별도 temp·반복 복원 검산](PROOF.json)

저장소 루트에서 실행한다.

```text
python research/informs_broad_validation_20261015_v1/stats_pack_numeric.py restore --manifest research/informs_broad_validation_20261015_v1/stats_packs/pack_numeric_postT_fixed_pca6_root_20261016_v1/MANIFEST.json --destination .
```

같은 원문 파일은 읽기만 하고 다른 기존 파일은 덮어쓰지 않는다. 동일 zlib 재현이 필요하며 SHA 검사를 생략하거나 공차로 대체하지 않는다. 새 FORMAT은 원 stats_pack/publication inspector를 변경하거나 위장하지 않는다. 최종 게시에는 별도 numeric inspect_archive 및 명시 allowlist 검사가 필요하다. 이 검산은 공개 SSE/SAE/count 집계의 저장 포맷 검산이며 새 학습·모델·시계열 원자료·정답 배열 평가가 아니다.
