# 과거 PeMS V 충분통계 원바이트 수동 복원

MANUAL_RESTORE_BEFORE_HISTORICAL_CONSUMERS

이 안내는 아래 단일 파일에 한해 과거 RESTORE_INDEX_KO.md의 parent_full_cpu 원공개경로 안내보다 우선한다. 과거 P2 감사 등 원 경로를 여는 소비자를 실행하기 전에 파일을 같은 원 경로로 수동 복원해야 한다. 현재 새 N5 감사·보고·그림의 공개 입력은 제거하지 않는다. 원 과학 보고서와 로컬 원파일은 보존한다.

```text
commit 1f5d34cb3b40de1a94368a45bfe37785d0fe7c5f
path research/informs_broad_validation_20261015_v1/priority_core_followup_20261016_v1/parent_full_cpu/diagnostic_pems08_B2_parent_full_CPU_V148_k43_s92601_v1/V_select/PUBLIC_ERROR_STATS.csv.gz
bytes 1661964
SHA256 72210e6872e3982159170c944d7619fbfb7a92ff9a2887181bb4ebc6a3ba4cd9
Git blob 4cf02af249e040e777cbd500a2a83edbe7b44118
```

[원 immutable 파일](https://raw.githubusercontent.com/CanelE452/hierarchical-tsfm-peft/1f5d34cb3b40de1a94368a45bfe37785d0fe7c5f/research/informs_broad_validation_20261015_v1/priority_core_followup_20261016_v1/parent_full_cpu/diagnostic_pems08_B2_parent_full_CPU_V148_k43_s92601_v1/V_select/PUBLIC_ERROR_STATS.csv.gz) · [바이트 검산](PROOF.json) · [manifest](MANIFEST.json) · [복원기](restore_git_v_stats.py)

복원기는 해당 object가 이미 있는 full-history clone 또는 managed worktree에서만 동작한다. .git가 파일인 worktree도 지원한다. GitHub ZIP, shallow/partial clone 또는 object가 없는 저장소에서 작동한다고 주장하지 않는다. fetch/network를 자동 실행하지 않으며 필요한 history는 사용자가 별도로 준비해야 한다. 다음 명령의 두 root는 저장소 root이다.

```text
python -B research/informs_broad_validation_20261015_v1/publication_derivatives/git_history_v_stats_20261017_v1/restore_git_v_stats.py --execute --root-triggered --repo-root . --destination-root . --manifest research/informs_broad_validation_20261015_v1/publication_derivatives/git_history_v_stats_20261017_v1/MANIFEST.json --manifest-sha fc4f7fbf226f721605cc0d55bf40159d2f3c97188b4ce1b949382accafb4f874 --source-sha 273837fae7ea7fdfa5eaa507e5f46ce7748b3d6c5a8fd0ce0472917f89c84a6c
```

원 gzip 헤더를 포함한 bytes/size/SHA256/Git OID를 확인하며 압축을 풀거나 재압축하지 않는다. 새 파일만 만들고 동일 파일은 skip, 다른 기존 파일은 거절한다. 경로 이탈과 symlink/reparse를 거절한다. 새 CSV 수치 해석·T 평가·모델 추론·과거 소비자 재실행의 검증은 아니다. 실제 공개 증분 상한 때문에 current selected tree에서 이 역사적 한 blob만 제외하며 이전 Git commit과 로컬 원파일의 원바이트는 유지한다. 전체 MASTER는 미완료다.
