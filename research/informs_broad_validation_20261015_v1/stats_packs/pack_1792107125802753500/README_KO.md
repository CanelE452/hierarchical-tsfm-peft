# 공개 충분통계 pack

Clone 직후 원 PUBLIC_ERROR_STATS.csv.gz 경로는 비어 있다. 아래 restore 명령을 실행해야 원 상대 경로와 원 gzip byte가 복원된다.

숫자·행·run ID·CSV byte를 보존한다. 같은 zlib 환경에서 원 gzip byte의 SHA를 별도 임시 디렉터리 복원으로 전수 확인했다. zlib가 다르면 원 gzip 재생성이 달라질 수 있으며 덮어쓰지 않고 실패한다.

- [원본 SHA·형식·archive manifest](MANIFEST.json)
- [공개 payload 목록](PAYLOAD_FILES.txt)
- [수정이 필요한 기존 Markdown 링크 목록](MARKDOWN_LINK_REPLACEMENTS.json)

저장소 루트에서 다음 명령을 사용한다.

```text
python research/informs_broad_validation_20261015_v1/stats_pack.py restore --manifest research/informs_broad_validation_20261015_v1/stats_packs/pack_1792107125802753500/MANIFEST.json --destination .
```

이미 일치하는 파일은 읽기만 하며, 불일치 파일을 덮어쓰지 않는다. 원본은 로컬에서 삭제하거나 변경하지 않았다.
