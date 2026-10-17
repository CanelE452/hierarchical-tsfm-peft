# 역사적 통계 pack 단일 chunk 원바이트 수동 복원

MANUAL_RESTORE_BEFORE_HISTORICAL_CONSUMERS

아래 단일 tar.xz에 한해 이 안내가 과거 pack README·REPRODUCE·RESTORE_INDEX의 직접 공개 경로 설명보다 우선한다. 기존 stats_pack.py 소비자를 실행하기 **전에 같은 원 경로로 수동 복원**한다. 나머지 archive·서명 manifest·원 과학 문서·로컬 원파일은 그대로 보존한다. 현재 N6 감사·보고·그림의 공개 입력을 제외하지 않는다.

```text
commit a32f88c184a5241585ca1fb50119f9462d91c402
path research/informs_broad_validation_20261015_v1/stats_packs/pack_1792097901794080200/part_0000.tar.xz
bytes 11900560
SHA256 b3f22f6675d5ca605440be7929cd2eeb10f3be38977798444c835352bc281a6a
Git blob 7b6ecb09c7c3106f31590998b10c93d8f832d5db
```

[immutable 원파일](https://raw.githubusercontent.com/CanelE452/hierarchical-tsfm-peft/a32f88c184a5241585ca1fb50119f9462d91c402/research/informs_broad_validation_20261015_v1/stats_packs/pack_1792097901794080200/part_0000.tar.xz) · [manifest](MANIFEST.json) · [바이트 검산](PROOF.json) · [복원기](restore_git_pack.py)

이미 해당 Git object가 있는 full-history clone 또는 managed worktree가 필요하다. GitHub ZIP·shallow/partial clone·누락 object에서 동작한다고 주장하지 않는다. 자동 fetch/network는 없다. 아래 두 root는 저장소 root이다.

```text
python -B research/informs_broad_validation_20261015_v1/publication_derivatives/git_history_pack_20261017_v1/restore_git_pack.py --execute --root-triggered --repo-root . --destination-root . --manifest research/informs_broad_validation_20261015_v1/publication_derivatives/git_history_pack_20261017_v1/MANIFEST.json --manifest-sha 30b5cdc7da170abc9d0ef8c86e39ae6dbcd1d015d0c209828db89b918b0ecf51 --source-sha 884ec087a905387ba432b6d7016cc69ddfff5db86efb5806ec9f2a599d962d9e
```

정확 commit:path·blob OID·크기·SHA256을 대사해 compressed 원바이트만 복원한다. tar/xz를 해제하거나 재압축하지 않는다. 새 파일은 생성하고 동일 바이트는 skip하며 다른 기존 바이트·경로 이탈·symlink/reparse는 거절한다. 복원 후 기존 pack 소비자의 원 size/SHA 검사와 기존 복원 절차를 따른다. 과거 소비자를 이번에 실행·통과했다고 주장하지 않는다.

실제 공개 용량 상한 때문에 current selected tree에서 이 역사적 한 chunk만 제외한다. 이전 immutable Git commit과 로컬 원본은 유지한다. 실제 바이트 검산은 별도 두 fresh 복원과 각각 repeat 및 첫 파일 mismatch sentinel에 한한다. 수치·보호 T 해독·새 평가·모델 추론 검증이 아니며 전체 MASTER는 미완료다.
