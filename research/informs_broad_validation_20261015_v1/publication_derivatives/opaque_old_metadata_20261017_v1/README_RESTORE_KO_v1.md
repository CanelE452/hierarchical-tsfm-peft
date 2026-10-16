# 공개 메타데이터 원바이트 복원

[확인] 이미 공개됐던 메타데이터 세 파일을 원바이트 그대로 gzip으로 보관한다. 원합계 2,905,599B에서 gzip 합계 291,385B가 됐다. 원 JSON/UTF-8/숫자는 해석하지 않았고 새 보호 T 수치 접근도0이다. 이는 게시 공간 확보이며 새로운 학습·추론 결과가 아니다.

[복원 manifest](GZIP_MANIFEST.json) · [실제 바이트 보존 검증](PROOF.json) · [동결 복원기](restore_originals.py) · [실행 코드](pack_metadata_v2.py) · [검사 코드](inspect_opaque_metadata_v2.py)

저장소 루트에서 다음을 실행하면 원 세 파일을 별도 폴더에 복원한다. 복원 폴더를 저장소 루트로 지정하면 없는 원 경로를 생성하며, 같은 바이트의 기존 파일은 건너뛰고 다른 바이트는 덮지 않고 거부한다.

```sh
python research/informs_broad_validation_20261015_v1/publication_derivatives/opaque_old_metadata_20261017_v1/restore_originals.py --destination restored_metadata
```

대상은 캠페인 루트의 과거 RESUME_QUEUE와 과거 multik 공개 운송 묶음 두 개의 BATCH_LAUNCHER_RECEIPT뿐이다. 정확한 경로·크기·SHA는 manifest에 있다. 과거 재개 큐는 역사 기록이며 자동 실행하지 않는다. 최신 우선순위·보호 T 잠금·원 마감과 현재 실제 완료 여부를 먼저 확인해야 한다.

실제 CPU2/RSS1GiB 외부 guard에서 압축 및 서로 독립인 새 private 두 폴더의 각3파일 복원, 반복 skip, 잘못된 기존 바이트의 덮기 거부를 수행했다. ROOT가 종료·현재 PID 생성시각 부재와 현재 여섯 복원 바이트 SHA를 다시 확인했다. 원 로컬 세 파일과 원 source39/immutable9는 보존됐다. 학습·비용 작업과 겹치지 않았다. private PLAN·자원/복원 영수증과 개인 경로는 공개하지 않는다.

[준비 당시 문서](README_PREPARATION_KO_v2.md)와 [초안](PLAN_DRAFT_v2.json)의 actual0 표시는 실행 전 스냅샷이다. 실제 실행 상태는 이 문서와 PROOF를 따른다. 별도 최종 게시 검사는 전체 selected 총량·상대 링크·Git tree/blob·원격 main을 확인한다. 전체 MASTER는 미완료이며 바이트 검사가 성능 개선·수렴·일반화 검증을 뜻하지 않는다.
