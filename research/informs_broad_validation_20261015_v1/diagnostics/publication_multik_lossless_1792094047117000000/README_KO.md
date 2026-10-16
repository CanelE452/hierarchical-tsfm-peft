# K 전환 비용 원본의 바이트 보존 게시 recipe

이 폴더는 기존 실측 비용의 게시 중복을 줄이기 위한 보조 코드입니다. 준비 시점에는 **합성 테스트만 실행**했습니다. 실제 K8 압축·1,440 trial 전수 검증·게시 후보 변경은 아직 하지 않았습니다. 해당 실제 실행은 별도의 `bundle_*` 폴더와 `BUILD_RESULT.json`이 있어야 확인할 수 있습니다.

원 측정 코드는 [measure_multik_transition.py](../../measure_multik_transition.py)의 `_json_new`, `measure`, `_durable_trial_check`입니다. 고정 writer SHA는 `db06213bbf1ac315022e76d7946d2b070f49dfd163c73ff20a7b6a2be36c85cb`입니다. 측정·예측·모델 코드를 import하거나 실행하지 않습니다.

## 유지하는 정보

- 원본 `TRIALS.csv` 8개와 나머지 queue·scope·batch 영수증은 바이트 그대로 복사합니다.
- `RESULT.json`·`NUMERIC_EXECUTION.json` 16개는 원본 바이트를 gzip으로 저장하고 원본과 gzip의 길이·SHA를 함께 기록합니다.
- trial 1,440개는 각각 원 상대경로·인덱스·바이트 수·SHA·참조 RESULT 경로를 기록합니다. `RESULT.rows`에서 복원한 모든 파일이 실제 원본과 바이트·SHA까지 일치해야 성공합니다.
- 각 scope의 12 transition ledger, 전체 20 scope 계획과 미실행 12 scope, 타이머·물리 F stream·메모리·variant·residency·출처 필드는 원 JSON 안에 모두 남습니다. 선택적 필드 삭제, 숫자 반올림, 행 축소를 하지 않습니다.
- 원본 파일은 삭제하거나 변경하지 않습니다. 후보에서 원 trial JSON을 제외할 수 있다는 판단은 실제 전체 복원 `PASS` 이후에만 가능합니다. 이 코드 자체는 후보 목록을 수정하지 않습니다.

현재 읽기 조사에서 원 cost 폴더는 1,490파일·7,730,387B이고, 그중 trial 1,440개가 1,986,622B였습니다. 실제 gzip 크기는 아직 측정하지 않았습니다. 전체 캠페인의 비용 완료를 의미하지 않으며 원 20 scope 중 실측 8 scope의 게시만 다룹니다.

## 실제 압축 실행 — 정밀 비용 측정과 겹치지 않을 때

저장소 루트에서 기존 프로젝트 Python을 `$Python`으로 지정한 뒤 실행합니다. `bundle_RUN_ID`는 아직 존재하지 않는 고유 폴더로 바꿉니다.

```powershell
$Campaign = (Resolve-Path 'research/informs_broad_validation_20261015_v1').Path
$Recipe = Join-Path $Campaign 'diagnostics/publication_multik_lossless_1792094047117000000'
& $Python -X utf8 -u -B (Join-Path $Recipe 'bundle_multik.py') prepare `
  --public-root $Campaign --output (Join-Path $Recipe 'bundle_RUN_ID')
```

CLI는 고정 writer SHA, 8 RESULT와 1,440 trial 개수를 검사합니다. 모든 원본을 먼저 snapshot한 뒤 trial 바이트·CSV·ledger를 검증합니다. 압축 후 별도 임시 경로에 전체 1,490파일을 실제 복원하여 모든 SHA를 확인하고, 같은 경로에서 재실행하여 이미 정확한 파일만 건너뛰는 동작을 확인합니다. 마지막으로 원본 전체 경로와 바이트가 그대로인지 다시 확인합니다. 어느 검사가 실패해도 성공 영수증을 쓰지 않습니다. 실패한 새 출력 폴더와 모든 원본은 그대로 보존됩니다.

## clone에서 원 경로 복원

```powershell
& $Python -X utf8 -u -B (Join-Path $Recipe 'bundle_multik.py') restore `
  --manifest (Join-Path $Recipe 'bundle_RUN_ID/MANIFEST.json') --public-root $Campaign
```

복원 위치는 campaign 내부의 원 `costs/multik_transition_1792090892017452100/`입니다. 모든 payload·출력 경로·SHA와 이미 존재하는 파일을 먼저 검사한 뒤 씁니다. 동일한 기존 파일은 건너뛰며 다른 기존 파일은 덮지 않습니다. 경로 탈출과 손상된 gzip은 출력을 쓰기 전에 거절합니다.

trial 복원은 OS의 기본 개행에 의존하지 않습니다. 다음 원 writer 형식을 명시적으로 재현합니다.

```python
text = json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
original_bytes = text.replace('\n', '\r\n').encode('utf-8')
```

원 dict 순서도 유지합니다(`sort_keys=False`). Python float를 다시 계산하거나 표시 정밀도를 줄이지 않습니다. 다른 Python 버전에서 encoder 출력이 달라지면 SHA 검사에서 실패하며, 근사 일치로 통과시키지 않습니다. gzip 헤더의 `mtime=0`은 새로운 gzip 파생물의 재현을 위한 설정이고, 복원되는 원 JSON 바이트와는 별개입니다.

## 합성 검증 범위

[test_bundle.py](test_bundle.py)는 합성 1 scope·2 trial만 사용합니다. 실제 비용 입력·시계열·모델·체크포인트·NPZ·T 값은 읽지 않습니다. 정확한 반복 복원과 원본 불변, trial 정밀도/CRLF/CSV/ledger 오류, 손상 payload, 경로 탈출, 기존 다른 파일 덮기 거절을 검사합니다. 이 검사 통과를 실제 1,440 trial의 전수 검증으로 바꾸어 말하지 않습니다.

```powershell
& $Python -X utf8 -u -B (Join-Path $Recipe 'test_bundle.py')
```

실제 게시 범위 검사와 상위 캠페인 보고·Git stage/commit/push는 별도 단계입니다.
