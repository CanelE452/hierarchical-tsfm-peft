# P2 공개 충분통계3개 무손실 표현

CSV 모든 행/순서/원문 token/개행과 원 gzip byte SHA를 보존한다. 원자료·모델·새 평가가 아니다. 원 로컬 파일은 유지한다.

[Manifest](MANIFEST.json) · [실제 복원 검산](PROOF.json)

저장소 root에서 복원:

```text
python research/informs_broad_validation_20261015_v1/stats_pack_numeric.py restore --manifest research/informs_broad_validation_20261015_v1/stats_packs/pack_numeric_priority_followup_20261016_v1/MANIFEST.json --destination .
```

같은 파일은 SHA검증 후 skip, 다른 기존 파일은 거부한다. frozen reader/FORMAT은 변경하지 않았다. 전체250MiB/게시 검사와 index 표현 대체는 ROOT의 별도 단계다.
