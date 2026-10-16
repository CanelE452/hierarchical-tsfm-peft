# 후속 fresh K 비용 포장 준비

이 폴더는 후속 비용의 **공개 포장 준비 코드**입니다. 현재 비용 폴더를 읽거나 열거하지 않았으며, 실제 압축·복원·원자료·배열·모델·GPU 실행은 하지 않았습니다. 준비 완료를 비용 측정 완료나 실제 2,160 trial 검산 완료로 해석하지 않습니다.

[package_fresh.py](package_fresh.py)는 [기존 K8 포장 코드](../publication_multik_lossless_1792094047117000000/bundle_multik.py)의 generic `build/restore`를 그대로 호출합니다. 기존 CLI의 8/1,440 고정만 피하고 원 build/restore 소스와 알고리즘은 변경하지 않습니다. 원 helper SHA는 `5632fa60313c41d62b1573ca67225cfb9ad9dadd7e494ba14c506388548a24dc`입니다. 따라서 파생 알고리즘의 AST 변경은 없습니다. thin caller는 별도 새 코드이며 실제 비용 worker로 실행된 버전이 아닙니다.

사전 조건과 실행 경계는 다음과 같습니다.

- 기본 실행은 정적 helper·writer·기존 자원 계획만 확인합니다. 현재 비용 폴더 stat/glob/read, framework import, 압축은 없습니다.
- `--execute-package --root-triggered`를 주어야 실제 포장을 시작합니다. ROOT가 실제 cost process terminal을 먼저 확인해야 합니다.
- 실행 시 명시적인 terminal RESULT 상대경로/SHA와 BATCH SHA가 필요합니다. CLOSED 중앙 영수증의 실제 SHA/필드, ACTIVE lease 부재, source39와 선택/등록표 보존 증거를 대조합니다. 자원 초과나 부분 상태를 포장 과정에서 성공으로 바꾸지 않습니다.
- 원 20개 큐의 순서·과학 필드는 유지하고 실행 후 달라지는 `status`만 사전 계획 상태와 구별합니다. 완료 재사용8은 기존 K8 MANIFEST의 scope별 모든 원 파일 SHA와 일치해야 합니다.
- 새 폴더의 fresh scope별 모든 파일 SHA를 actual receipt와 대조합니다. 실제 durable RESULT 개수와 실제 trial 합계를 API에 전달하며, 계획 최대12/2,160을 실측으로 쓰지 않습니다.
- root BATCH, FULL_QUEUE_PLAN, CSV와 모든 JSON을 원바이트 그대로 보존합니다. RESULT/NUMERIC은 gzip `mtime=0`, trial은 원 Windows CRLF 직렬화로 바이트·SHA를 전수 복원합니다. 원 숫자/순서/정밀도를 변경하지 않습니다.
- generic build가 별도 임시 폴더에 모든 원 파일을 실제 복원하고 같은 폴더에서 다시 복원한 뒤 원본 전체 경로·바이트 불변을 검사합니다. 기존 다른 파일은 덮지 않습니다.
- RESULT/NUMERIC/CSV가 없는 비정상 중단, orphan trial, 비연속 번호, JSON/CSV 이외 파일은 원 helper가 거절합니다. 해당 원본과 실패 출력은 보존하며 임의 생략하거나 다른 조건으로 대체하지 않습니다.

ROOT가 terminal과 CPU 창을 확인한 뒤 사용할 명령 형식입니다. 아래는 준비 예시이며 아직 실행하지 않았습니다.

```powershell
$Recipe = Join-Path $env:INFORMS_PUBLIC_ROOT 'diagnostics/publication_multik_fresh_lossless_20261016_v1'
& $Python -X utf8 -B (Join-Path $Recipe 'package_fresh.py')

# ROOT가 terminal 증거와 최종 BATCH SHA를 확정한 뒤에만 실행
& $Python -X utf8 -B (Join-Path $Recipe 'package_fresh.py') `
  --execute-package --root-triggered `
  --terminal-result 'diagnostics/<실제 terminal 폴더>/RESULT.json' `
  --terminal-sha256 '<실제 terminal RESULT SHA>' `
  --batch-sha256 '<실제 최종 BATCH SHA>' `
  --output 'diagnostics/publication_multik_fresh_lossless_20261016_v1/bundle_<새 고유 ID>'
```

포장 후에는 새 bundle의 README/INDEX·MANIFEST·BUILD_RESULT와 payload만 게시 후보에 추가합니다. fresh 원 trial/RESULT 경로를 별도 게시 후보에 추가하지 않으며, 기존 K8 bundle은 별도 그대로 유지합니다. 원 경로 목록과 원 SHA는 복원 MANIFEST에 남고 Markdown은 bundle 문서로 연결합니다. 전수 실측 포장이 성공한 뒤에만 실제 수·용량·복원 PASS를 후속 영수증에 기록합니다.

현재 검증 범위는 원 helper의 작은 합성 7개 검사, 새 caller AST/default metadata, 실행 인자 부족 거절입니다. 실제 fresh 데이터 검산과 전체 실행 경계 검증은 아직 하지 않았습니다.
