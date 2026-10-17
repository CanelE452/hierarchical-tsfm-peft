# 과거 SVG 8개의 Git history 바이트 보존

기존 공개 그림 8개(1,443,017B)는 원 로컬 파일과 main `b76a8384d0e870c6e6aed365f6b52889d1c7b035`의 Git blob에 그대로 보존한다. 현 [그림 색인](../../PUBLIC_FIGURES_KO.md)의 해당 8개 URL만 그 commit의 immutable raw URL로 연결한다. 파일을 재압축하거나 그림·보호 T의 수치를 다시 해석하지 않았다.

[원 경로·길이·SHA256·Git OID](MANIFEST.json) · [실제 바이트 복원 검산](PROOF.json) · [복원 도구](restore_git_history_svg.py) · [이번 실험 보고](../../priority_multik_resume_20261016_v1/MULTIK_RESUME_KO_v1.md)

[확인] 실제 로컬 Git 객체로 서로 다른 두 새 목적지에 각각 8파일을 복원하고, 반복 호출에서는 8파일 모두 동일 바이트라 건너뛰었다. 별도 첫 파일 sentinel의 불일치는 덮어쓰기 없이 거부했다. 공개 proof는 원본·복원본의 길이/SHA256 및 Git blob OID, source48 전후 보존, 원 마감 내 실제 검산을 결속한다. 모든 파일의 원자적 rollback이나 전체 repository fsck를 주장하지 않는다. 파일별 CreateNew이므로 뒤 파일에서 실패하면 앞서 정상 복원한 파일은 남는다.

재현에는 **원 commit과 선택 blob을 가진 full-history clone 또는 managed worktree**가 필요하다. shallow/partial/promisor clone과 alternate object store는 도구가 거부하며, ZIP만으로는 Git 객체 복원을 재현할 수 없다. 도구는 fetch·lazy fetch를 하지 않는다. 전역·시스템 Git 설정과 replace object를 사용하지 않고 위험한 local include/promisor 구성은 거부한다. 현재 원 경로에 파일이 존재하면 동일 바이트는 건너뛰고 다른 바이트는 덮지 않는다.

저장소 root에서 수동 복원할 때:

```sh
python -B research/informs_broad_validation_20261015_v1/publication_derivatives/git_history_svg_20261017_v1/restore_git_history_svg.py --execute --root-triggered --repo-root . --destination-root . --manifest research/informs_broad_validation_20261015_v1/publication_derivatives/git_history_svg_20261017_v1/MANIFEST.json --manifest-sha 15308425d747a8b45d43f5083d18dc80102e8df4aff77325a3d4473d9c2990fe --source-sha 98b9161944ed9f86c62e15ed8d79372d337583e666872c15b2a51a11b4664a0b
```

기본 호출은 입력 읽기·Git·쓰기 0이다. 위 수동 도구는 campaign 이후에도 재현할 수 있다. 이번 실제 ROOT 검산은 별도의 CPU2/RSS3GiB supervisor와 `--worker-deadline 1792222550`을 사용했으며 원 실험 자원을 연장하지 않았다. private fresh 목적지와 private 실행 경로는 공개하지 않는다. HTTP 검산은 게시 직후 별도로 수행하며, 바이트 일치가 GitHub inline 렌더·레이아웃 확인이나 모델 성능의 입증을 대신하지 않는다. 전체 MASTER는 미완료다.
