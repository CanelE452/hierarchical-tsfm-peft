# 기존 충분통계 archive의 원바이트 복원

기존 V 충분통계의 part 하나를 무손실 숫자 인코딩으로 다시 포장했다. 기존 로컬 `part_0000.tar.xz`와 원 MANIFEST는 바꾸거나 삭제하지 않았다. 새 포장은 원 member16개, 순서·tar header/padding 및 **원 XZ archive12,132,156byte의 SHA**까지 동일하게 재생성한다. 원 archive의41개 record에 해당하는 집계 행은 정밀도·개행·행 수를 유지한다. 새 모델 학습·추론·TEST target 평가를 수행한 작업이 아니다.

- [새 transport manifest](MANIFEST.json), [원 archive의 별도·반복 복원 proof](PROOF.json), [공개 복사 SHA](ROOT_COPY_BINDING.json), [ROOT의 별도 실제 CLI 복원](ROOT_INDEPENDENT_RESTORE.json)
- [기존의 원 MANIFEST](../../stats_packs/pack_1792065325663656600/MANIFEST.json)는 그대로 유지한다. part1은 [별도 transport](../transport_legacy_V_part1_20261016_v1/README_KO.md)로 보존하고 나머지9개 archive part는 같은 경로에 유지한다.

새 clone의 저장소 루트에서 **이 transport를 먼저 복원**한 뒤 원 충분통계를 복원한다.

```text
python research/informs_broad_validation_20261015_v1/stats_pack_transport.py restore --manifest research/informs_broad_validation_20261015_v1/publication_derivatives/transport_legacy_V_part0_20261016_v1/MANIFEST.json --destination .
python research/informs_broad_validation_20261015_v1/stats_pack_transport.py restore --manifest research/informs_broad_validation_20261015_v1/publication_derivatives/transport_legacy_V_part1_20261016_v1/MANIFEST.json --destination .
python research/informs_broad_validation_20261015_v1/stats_pack.py restore --manifest research/informs_broad_validation_20261015_v1/stats_packs/pack_1792065325663656600/MANIFEST.json --destination .
```

첫 명령은 원 `stats_packs/pack_1792065325663656600/part_0000.tar.xz`만 만든다. 두번째 명령은 원 MANIFEST와 나머지 parts를 사용해 원 PUBLIC_ERROR_STATS.csv.gz 경로들을 복원한다. 같은 기존 파일은 건너뛰며 불일치 파일은 덮어쓰지 않는다. 원 XZ byte 재현에 맞는 liblzma, 원 gzip byte 재현에 맞는 zlib 환경이 필요하다. 다르면 SHA 공차를 완화하지 않고 실패한다.

원 archive12,132,156byte에서 새 payload6,849,268byte로5,282,888byte를 줄였다. 이는 sidecar를 제외한 수치이며 최종 게시 용량은 sidecar와 전체 선택 파일을 합쳐 별도 검사한다. Git의 과거 commit/object를 삭제하거나 history 크기를 줄인 작업은 아니다. 새 FORMAT의 검사는 [numeric/transport 공개 검사기](../../publication_check_numeric.py)를 사용하며, 원 publication inspector가 이 형식을 지원한다고 주장하지 않는다.

포장·복원 PASS는 수치 보존 검사다. 연구 성공·수렴·독립 일반화·원 연구 전체 완료·main push를 뜻하지 않는다. 전체 복원 순서는 [공개 복원 색인](../../RESTORE_INDEX_KO.md)을 따른다.
