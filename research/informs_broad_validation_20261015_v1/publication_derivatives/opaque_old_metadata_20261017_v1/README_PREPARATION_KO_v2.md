# 기존 공개 메타데이터 세 파일의 바이트 보존 게시 준비

**PREP_ONLY. 실제 원파일 압축·복원·검사·게시를 실행하지 않았다.** 현재 두 fit과 실제 실행된 모든 후처리의 강한 종료·자원 반환을 ROOT가 직접 확인해야 한다. 학습·검산·비용의 과학적 성공은 이 바이트 표현 전환의 입장 조건이 아니다. 실패·예산 제한·건너뛴 경로는 해당 상태를 그대로 보존한다. 이 준비물은 실험 결과나 저장 공간 절감의 실측치가 아니다. v1 여섯 파일과 기존 복원기는 변경하지 않았다.

[초안](PLAN_DRAFT_v2.json)의 정확한 세 파일만 대상이다. 원 SHA와 크기는 기존 공개 selected inventory에서 읽었으며, 이번 준비에서는 원파일 본문을 읽지 않았다. 최신 N3/N4, P1/P2, 원수치 MD/CSV/SVG, source39/immutable9, 모델·데이터·예측 배열은 새 압축 대상으로 추가하지 않는다. 원파일은 로컬에 보존하고, Git index에서의 표현 전환은 실제 검증을 마친 뒤 ROOT가 별도로 수행한다.

[pack_metadata_v2.py](pack_metadata_v2.py)는 원파일을 JSON·UTF-8·숫자로 해석하지 않고 바이트 그대로 deterministic gzip으로 저장하도록 준비했다. 기존 `exact_auxiliary_gzip_v1` manifest와 [원바이트 그대로 복사한 기존 복원기](restore_originals.py)를 재사용한다. 실제 실행 시 원파일·source48 입력의 실행 전후 SHA, 서로 다른 새 private cache 두 곳의 실제 복원, 각 반복 호출의 skip, 다른 기존 바이트의 덮기 거부를 확인한 뒤에만 새 `PROOF.json`을 작성한다. 실패한 중간 출력은 보존하며 성공으로 표시하지 않는다.

[inspect_opaque_metadata_v2.py](inspect_opaque_metadata_v2.py)는 정확한 새 namespace의 `files/item_0000.opaque.json.gz`부터 `0002`까지 세 파일에만 한정된다. 기존 selected inventory의 원 SHA·크기, manifest, 실제 복원 PROOF의 ROOT 지정 SHA에 묶어 압축 바이트의 SHA·크기만 확인한다. 이 경로에서는 gzip 해제나 원 JSON/UTF-8/보호 T 숫자 파싱을 하지 않는다.

동결된 `publication_check_numeric.py`는 변경하지 않는다. 별도로 로드한 numeric 모듈의 dependency 반환값에 좁은 facade를 두어 세 파일을 동결 `pc.inspect(..., virtual_files=...)`에 전달하고, 나머지 변경·신규 파일은 원래 numeric 검사에 그대로 보낸다. 원래 링크 검사도 유지한다. wrapper는 세 파일을 포함한 명시 목록의 250MiB 합계와 파일별 20MiB를 검사한다. **전체 최종 selected inventory, 생성 manifest를 포함한 cap, Git index·tree·blob, 상대 링크, 원격 SHA 검증은 ROOT의 다음 게시 게이트가 별도로 수행해야 한다.** wrapper PASS가 전체 저장소나 과학적 결과의 새 검증을 의미하지 않는다.

실제 실행은 아직 금지되어 있다. ROOT가 소스 전체 읽기와 독립 리뷰 후 새 `PLAN`을 봉인해야 하며, 초안의 `ROOT_admitted=false`와 null SHA로는 실행할 수 없다. 새 PLAN에는 검토된 v2 runner/inspector SHA, 실제 source39·immutable9, 기존 inventory 경로, private cache 경계와 새 quiet admission의 경로·SHA가 필요하다. source39·immutable9의 키는 저장소 기준 `research/informs_broad_validation_20261015_v1/`로 시작해야 한다. 실제 private PLAN과 quiet/closure 영수증은 개인 절대경로를 포함하므로 공개하지 않는다. 공개 PROOF에는 SHA 및 과학적 상태의 제한된 메타데이터만 싣는다. 기본 CLI는 인자를 생략해도 입력 읽기·출력 쓰기가 0이다.

quiet admission의 schema는 `ROOT_OPAQUE_METADATA_QUIET_ADMISSION_v2`이다. ROOT는 새 현재 프로세스 점검을 완료한 시각, `actual_current_process_inspection=true`, `actual_current_campaign_CIM_empty=true`, 현재 캠페인의 모델·비용·압축 프로세스가 없다는 결과와 실제 receipt descriptor를 기록한다. 각 descriptor는 실제 파일의 `path`와 `sha256`을 갖는다. `fit_dispatch_result`는 정확한 두 run의 `COMPLETE_TWO_FITS_AND_FULL_V` 또는 `PARTIAL_TWO_FITS_PRESERVED` 결과다. `post_result`는 같은 fit dispatch SHA에 묶인 `COMPLETE_TWO_VALIDATIONS_AND_FOUR_COSTS` 또는 `PARTIAL_VALIDATIONS_COSTS_PRESERVED` 결과다. 두 결과의 error는 null, pins unchanged는 true여야 한다. 과학적 실패·부분 완료 상태를 그대로 기록하며 양쪽 성공은 요구하지 않는다.

`fit_guard_binding`은 실제 fit `spec`, `guard`, ROOT `closure` descriptor를 갖는다. `executed_post_guard_bindings`에는 실제 post `steps`에 등장한 id마다 같은 세 descriptor와 `id`를 정확히 한 개씩 넣는다. 건너뛴 작업에 새 guard를 만들어 넣거나 실행된 guard를 누락할 수 없다. fit SPEC은 `3faf463d…765390`, 기존 helper는 `82a837ea…c75f8b`의 실제 SHA에 고정된다. 모든 실행된 worker의 관찰 tree join, errors/remaining 없음, root 및 owned PID/create_time을 확인한다.

ROOT closure schema는 `ROOT_OPAQUE_METADATA_OWNER_ABSENCE_v2`, status는 `VERIFIED_STRONG_OBSERVED_TREE_CLOSURE_AND_CURRENT_ABSENCE`다. 실제 `guard_sha256`, `spec_sha256`, `actual_PID_creation_time_check=true`, `remaining_exact_owners=[]`, `errors=[]`, `checked_unix`와 실제로 현재 부재를 확인한 `absent_identities`를 기록한다. 이 목록은 실제 owned identity를 `{pid, create_time}`으로 투영하여 `(pid, create_time)` 순서로 정렬한 것과 정확히 같아야 한다. 과거 로그만으로 현재 부재를 추정하지 않는다. 현재 존재하지 않는 영수증을 미리 작성하거나 합성 테스트를 실제 증거로 쓸 수 없다.

같은 admission에 `cpu_threads=2`, `RSS_limit_bytes=1073741824`, `worker_deadline_unix=1792222550`, 원 storage cap·reserve와 측정한 사용량을 기록한다. 최초·마지막 검증 시 admission이 120초 이내여야 한다. 이 Python 준비물은 프로세스를 직접 조회하거나 제어하지 않는다. CPU2/RSS1GiB·현재 quiet 상태·worker 마감의 실제 강제는 기존 ROOT supervisor가 맡아야 한다. 저장 cap 30GiB와 reserve 256MiB에 보수적인 추가 32MiB 여유가 없으면 실행하지 않는다. 원 worker 마감은 연장하지 않는다.

ROOT가 모든 전제를 실제 확인한 이후의 호출 형식은 다음과 같다. 이 문서는 실행 승인이 아니다.

```text
python -B <이 폴더>/pack_metadata_v2.py --execute --root-triggered --repo-root <기존 worktree> --private-restore-root <새 private cache> --plan <새 ROOT 봉인 PLAN> --plan-sha <그 PLAN의 전체 SHA>
python -B <이 폴더>/inspect_opaque_metadata_v2.py --execute --root-triggered --repo-root <기존 worktree> --file-list <명시 delta 목록> --file-list-sha <목록 SHA> --tracked-list <ROOT가 확인한 공개 경로 목록> --tracked-list-sha <목록 SHA> --old-inventory <기존 v5 inventory> --proof-sha <실제 PROOF SHA>
```

[작은 stdlib 합성 검사](test_opaque_metadata_v2.py)는 기본 0-I/O, ROOT trigger/초안/마감 거부, 정확한 세 원 SHA 존재, traversal·덮기·변조 거부, 기존 복원기 재사용, proof 범위, facade의 나머지 검사 위임과 loader 복원을 확인한다. v2 추가 검사는 한 fit이 실패해도 실제 자원이 모두 종료된 합성 조건의 입장, active·누락 closure·잘못된 SPEC·허구의 skipped guard 거부를 확인한다. 합성 receipt는 실제 실험 영수증이나 예상 실제 결과가 아니다. 이 검사는 실제 세 원파일, 현재 실험 결과, 실제 quiet admission, 전체 게시 cap 또는 원격 게시를 검증하지 않는다.
