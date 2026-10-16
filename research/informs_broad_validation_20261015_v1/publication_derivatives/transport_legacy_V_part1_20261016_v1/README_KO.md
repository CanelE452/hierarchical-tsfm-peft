# 기존 V 충분통계 part1의 원바이트 복원

기존에 공개한 V 충분통계 archive part1을 모든 행·정밀도·CSV/member/tar·원 XZ byte까지 보존하는 별도 FORMAT으로 다시 포장했다. 원 로컬 archive와 원 MANIFEST는 바꾸거나 삭제하지 않았다. 모델·관측 원자료·정답 배열·추론·학습에 접근한 작업이 아니다.

[새 MANIFEST](MANIFEST.json), [별도·반복 복원 proof](PROOF.json), [공개 복사 SHA](ROOT_COPY_BINDING.json), [ROOT 별도 CLI 복원](ROOT_INDEPENDENT_RESTORE.json)을 확인한다. part1에 연결된 기존 record20개의 원 metadata split은 모두 V_select였다. 원 member13개와 tar65,607,680byte를 보존한다.

원 archive12,006,828byte(SHA e79d2aed3b5ced0fbef6fbffa29a94af567fc764ea257e60e6df3cb82f66cf30)에서 새 payload6,539,876byte로 archive-only5,466,952byte를 줄였다. JSON/README sidecar와 최종 공개 전체 용량은 별도 계산한다. Git의 이전 commit/object를 지우거나 history 용량을 줄인 작업은 아니다.

새 clone 저장소 루트에서 아래 transport 두 개를 먼저 복원한 후 원 충분통계를 복원한다. 두 transport와 [원 MANIFEST](../../stats_packs/pack_1792065325663656600/MANIFEST.json)는 형식이 다르다.

```text
python research/informs_broad_validation_20261015_v1/stats_pack_transport.py restore --manifest research/informs_broad_validation_20261015_v1/publication_derivatives/transport_legacy_V_part0_20261016_v1/MANIFEST.json --destination .
python research/informs_broad_validation_20261015_v1/stats_pack_transport.py restore --manifest research/informs_broad_validation_20261015_v1/publication_derivatives/transport_legacy_V_part1_20261016_v1/MANIFEST.json --destination .
python research/informs_broad_validation_20261015_v1/stats_pack.py restore --manifest research/informs_broad_validation_20261015_v1/stats_packs/pack_1792065325663656600/MANIFEST.json --destination .
```

두 첫 명령은 원 part_0000.tar.xz와 part_0001.tar.xz를 만든다. 나머지9개 원 part는 기존 경로에 게시한다. 원 preset6/CRC64 XZ byte에는 재현 가능한 liblzma, 원 충분통계 gzip에는 같은 zlib 환경이 필요하다. SHA가 다르면 공차를 완화하거나 덮어쓰지 않고 실패한다. 일치하는 기존 파일은 읽고 건너뛴다.

검사는 [명시 numeric/transport 검사기](../../publication_check_numeric.py)의 두 정확한 transport namespace 허용 범위에서 수행한다. 원 frozen publication checker가 새 FORMAT을 지원한다는 뜻은 아니다. 포장/복원 PASS를 연구 성공·수렴·추가 실험·독립 일반화·main 게시 또는 원 연구 전체 완료로 해석하지 않는다. [전체 복원 순서](../../RESTORE_INDEX_KO.md)를 따른다.
