# 공개 파일·링크·Git index 검사

`publication_check.py`는 현재 campaign의 **공개할 파일을 개별 경로로 확정하고 그 바이트를 검증**한다. 데이터 재다운로드, 학습 재실행, 과학적 결과 검산을 대신하지 않는다. Git에는 조회 명령만 사용하며 stage, commit, push, 삭제, 기존 파일 덮어쓰기를 수행하지 않는다.

## 허용 목록과 제외 범위

저장소 루트는 campaign 디렉터리의 두 단계 상위이다. 파일명은 이 루트 기준 POSIX 상대 경로 한 줄씩 저장한다. 기본 후보 탐색은 campaign 내부로 제한하며, 저장소 README나 기존 history 변경 파일은 `--extra` 또는 `--file-list`로 각각 명시해야 한다. 기존 HEAD에 있는 파일은 링크 대상으로 사용할 수 있지만 자동으로 새 게시 묶음에 추가하지 않는다.

허용 확장자는 Python·PowerShell·shell 소스, Markdown, JSON/JSONL, CSV, gzip UTF-8 텍스트, PNG, SVG, YAML, TOML, TXT, SHA256이다. cache, private, raw, weights, checkpoints, `__pycache__` 디렉터리와 나머지 확장자는 제외한다. 따라서 NPZ/NPY, 모델 가중치, pickle, 로그, 임시 파일, 원 배포 ZIP은 포함하지 않는다. gzip은 압축을 풀어 UTF-8 및 민감정보 검사를 수행한다.

다음 원본은 로컬에 보존하고 공개 목록에서 제외한다.

- `plan/MASTER_PLAN.md`
- `plan/PACKAGE_MANIFEST.sha256`
- `plan/PACKAGE_README.md`
- `reports/snapshot_20261015T095115Z_1792057875518554100/` 전체
- 검사용 `checks/` snapshot 전체

공개 계획은 [MASTER_PLAN_PUBLIC.md](plan/MASTER_PLAN_PUBLIC.md)와 [PUBLIC_PLAN_PROVENANCE.json](plan/PUBLIC_PLAN_PROVENANCE.json)이다. JSON의 원본 근거 경로는 로컬 provenance로 남길 수 있다. **Markdown 링크·이미지와 달리 모든 JSON 경로가 공개 대상이어야 한다고 강제하지 않는다.** 추가로 제외할 과거 snapshot은 `--exclude-prefix`에 campaign 기준 상대 디렉터리를 넣는다. 명시적 `--file-list`는 검토자가 최종 파일을 직접 선택하므로 기본 후보의 추가 prefix 필터를 자동 적용하지 않는다.

## 바이트와 링크 검사

각 파일은 20 MiB, 공개 증가량은 250 MiB 이하이어야 한다. 증가량을 과소 계산하지 않도록 선택 파일의 전체 바이트 합을 **공개 증가량의 보수적 상한**으로 사용한다. 실제 변경 없는 기존 파일까지 일부러 선택하면 이 상한에 포함되며, 그런 파일은 링크로 참조하는 편이 맞다. 최종 생성 메타데이터 세 파일도 크기 제한에 포함한다.

파일 경로의 절대 경로·상위 경로 탈출·symlink를 거절한다. Markdown의 inline/reference 링크와 HTML href/src, SVG href는 URL 디코딩 후 저장소 밖으로 나가는지 검사한다. 로컬 대상은 실제 존재하고 이번 허용 목록 또는 기존 HEAD에 있어야 한다. 이미지의 PNG signature, SVG XML 루트, SVG script/event-handler도 검사한다. 외부 HTTP(S) 링크의 네트워크 도달 여부는 이 도구의 검사 범위에 포함하지 않는다.

UTF-8 문서에서 개인 절대 경로, 한국 휴대전화 형태, 실제 형태의 API/token/private-key 및 충분히 긴 secret 문자열 대입을 검사한다. 실패 기록에는 상대 파일명·규칙·행만 남기며 일치한 민감 문자열 자체는 저장하지 않는다. 소수, 긴 hash, Unix 초, FLOP·바이트 정수, 경로 정규식 리터럴 및 `<workspace>` 같은 자리표시자는 연락처·개인 경로로 오인하지 않도록 테스트한다. 패턴 검사는 모든 기밀의 부재를 수학적으로 증명하지 않으며, 검토된 공개 파일 범위 선택과 함께 사용한다.

## 실행 순서

현재 작업 상태에서는 preflight만 실행한다. 아래 명령은 모두 campaign 디렉터리 기준이다.

```powershell
python -X utf8 publication_check.py --preflight
python -X utf8 publication_check.py --preflight --file-list reviewed_payload_files.txt
```

preflight는 매번 새로운 `checks/preflight_<ns>/PREFLIGHT.json`만 생성한다. 실패가 있으면 원본을 그대로 보존한 채 보고하고, 다른 작업자의 파일을 자동 수정하지 않는다. 실행 중 기록이 바뀔 수 있으므로 이 snapshot을 최종 게시 확정이라고 부르지 않는다.

실험·보고서 쓰기가 끝나고 마지막 파일 목록이 확정된 뒤, root가 다음을 실행한다.

```powershell
python -X utf8 publication_check.py --prepare --file-list reviewed_payload_files.txt
python -X utf8 publication_check.py --verify
```

`reviewed_payload_files.txt`는 저장소 상대 경로의 **payload 파일만** 포함한다. 최종 생성할 세 메타데이터 파일은 코드가 자동으로 추가한다. 목록 자체를 게시하려면 그 파일도 payload에 명시한다. `--prepare`는 깨끗한 preflight와 직전 payload SHA 재확인 후 아래 파일을 exclusive-create 한다. 하나라도 이미 있으면 덮어쓰지 않고 중단한다.

- `PUBLIC_ALLOWLIST.txt`: payload와 생성 파일 모두의 정확한 개별 경로
- `PUBLICATION_CHECK.json`: payload별 SHA256·크기, 링크 검증, 허용 목록 SHA, 준비 시각
- `PUBLIC_MANIFEST.sha256`: **자기 자신만 제외**하고 payload·허용 목록·검사 receipt의 SHA256

receipt는 payload와 허용 목록을 결속하며 receipt 자신이나 manifest를 hash하지 않는다. 따라서 자기 hash 순환이 없다. 생성 도중 외부 오류로 일부 파일만 생기면 자동 삭제·재작성을 하지 않는다. 검토자가 기존 생성물을 보존한 뒤 재게시 위치를 정해야 한다. 준비 후 파일이 바뀌면 `--verify`의 manifest 또는 preflight payload 대조가 실패한다.

root가 허용 목록의 변경 파일을 **개별 경로로** stage한 후 실행하는 마지막 검사는 다음과 같다.

```powershell
python -X utf8 publication_check.py --staged
```

이는 먼저 디스크의 최종 manifest를 검증하고, `git diff --cached --name-only -z`의 정확한 범위를 허용 목록과 비교한다. 모든 index blob은 `git show :<relative-file>`로 읽어 디스크 바이트와 동일해야 한다. 외부 파일의 기작성 stage, 빠진 파일, stage 이후 디스크 변경은 실패한다. 변경 없는 HEAD 파일은 실제 staged diff에 등장하지 않으므로 게시 payload로 선택하지 않는다. 검사 통과가 push 수행을 의미하지 않는다.

## 검증

임시 디렉터리 CPU 테스트는 정상 prepare/verify, 재생성 금지, self-hash 제외, 경로·인코딩 탈출, 제외된 로컬 원본 링크, 잘못된 PNG/SVG, 파일·묶음 크기, 민감 문자열 비노출, 실제 연락처와 과학적 수치 구분, payload/허용 목록 변조, mocked index 범위·바이트 불일치를 확인한다. 테스트는 실제 Git index를 변경하지 않는다. symlink 테스트는 운영체제 권한상 생성할 수 있을 때 수행한다.
