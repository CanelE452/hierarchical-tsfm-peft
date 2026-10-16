# 제공된 공개 검산기 원본의 thin caller

**PREPARED — 실제 연구 V/T gzip 검산은 아직 실행하지 않았다.** ROOT가 최종 보고서·whole audit·비용 종료를 확인하고 명시 실행 신호를 준 뒤에만 `--execute-audit`를 사용한다.

제공 원본 `plan/audit_public_bundle.py` 전체를 읽었고 수정 없이 다음 SHA에 고정했다.

```text
7c5f705770361e81f64c0963245013b2b54d50bcebab6b7c7278df25389fbc2c
```

기본 메타데이터 결속은 기존 `public_horizon_summary_20261016_v1/horizon_summary.py`의 `metadata_plan`을 그대로 사용한다. 해당 코드 SHA는 `ca7d4711fca27c1dcc0903945b3d915f6a016279824736a19c29d58beaa9d9d2`다. 기존 helper·source39·registry·lock·점수·원 충분통계를 변경하지 않는다.

## 입력과 실행 범위

실제 최종 보고서의 `SCORES.csv`와 같은 보고서의 `INPUT_MANIFEST.json`을 명시한다. `scope=public` 입력 SHA가 각 score의 `manifest_relative_path`, 공식 manifest의 원 `PUBLIC_ERROR_STATS.csv.gz`를 결속해야 한다. 최종 표에 있는 V_select/T_final **전체 행 수 그대로** 처리하며 좋은 조건만 선택하지 않는다. 현 ROOT의 예상451은 실제 표를 읽어 확인할 값이지 실행 없이 완료로 쓰는 상수가 아니다.

제공 원본은 `(run_id,split,cohort)` 중복 score key를 거부한다. 중복을 가짜 ID로 분할하거나 이름을 바꾸거나 행을 빠뜨려 우회하지 않는다. M/E0 등 비표준 split이 최종 점수표에 섞이면 임의 제외 대신 실패한다. T는 기존 metadata helper가 실제 잠긴 허용 ID를 확인하며, 이 caller는 T 선택·새 추론을 하지 않는다.

기본 모드는 작은 CSV/JSON·source SHA와 gzip 크기만 확인한다. **gzip을 열거나 SHA 계산하지 않으며**, 임시 flat bundle·결과 snapshot·원본 audit 실행은0이다.

## 명시 실행 후의 임시 flat bundle

`--execute-audit`에서만 다음을 순서대로 수행한다.

1. 최종 SCORES 원바이트를 별도 `TemporaryDirectory`의 `SCORES.csv`에 복사하고 SHA를 확인한다.
2. 각 원 gzip을 `PUBLIC_ERROR_STATS/<index>_<fullSHA>.csv.gz`에 바이트 그대로 복사해 SHA·크기를 전수 확인한다. 같은 내용처럼 보여도 임의 dedup하지 않는다.
3. 임시 `PUBLIC_MANIFEST.sha256`에 SCORES와 **모든** gzip의 SHA를 기록한다. manifest는 자기 자신을 포함하지 않는다.
4. 제공 원본의 `audit(flat_root)`를 수정·대체 없이 한 번 호출한다. 원본이 숫자·origin/channel/horizon 전체 coverage·SAE/SSE Cauchy bound·채널 macro·flat manifest를 검사한다. checked score/stat 파일 수는 실제 입력 행 수, 검증 hash 수는 그 수+1이어야 한다.
5. source39, registry, lock, caller/helper, 원 SCORES·입력 영수증·공식 manifest·원 gzip의 사후 SHA를 대조한다. 임시 폴더는 자동 정리한다.

원본 검산의 숫자 PASS와 사후 불변 검사를 구분한다. 사후 실패는 원본 numeric 결과와 완료 copy ledger를 보존하면서 `INCOMPLETE_POSTFLIGHT`로 남긴다. 복사·원본 audit 실패는 `FAIL_PROVIDED_AUDIT_OR_COPY`, 실제 성공은 `PASS_PROVIDED_ORIGINAL_AUDIT`다. 실패 status는 CLI exit1이다.

제공 원본은 전체 표의 통계를 동시에 메모리에 보유한다. 실제451 범위의 실행 시간·메모리 상한은 합성 시험만으로 보장하지 않는다. 큰 실제 실행은 ROOT의 기존 자원 감시·실행 신호 아래 진행하며, 이를 피하려고 표를 나누거나 비교 조건을 줄이지 않는다.

## CLI와 출력

저장소 루트에서 `$Campaign`과 실제 최종 `$FinalReport`를 정한 뒤 사용한다. 준비되지 않은 최종 파일을 추정해 호출하지 않는다.

```powershell
$Tool = Join-Path $Campaign 'diagnostics/provided_public_audit_entry_20261016_v1/provided_audit_entry.py'
python -X utf8 -B $Tool --public-root $Campaign `
  --scores (Join-Path $FinalReport 'SCORES.csv') `
  --input-manifest (Join-Path $FinalReport 'INPUT_MANIFEST.json')
```

기본값은 metadata-only다. ROOT 실행 신호 후 같은 명령에 `--execute-audit`만 추가한다. 새 `snapshot_<ns>`에는 다음 파일을 남기며 기존 결과를 덮지 않는다.

- `RESULT.json`: 원본 audit 반환값·실제 행/복사 수·실패·사후 불변 상태를 분리한다.
- `INPUT_BINDINGS.json`: 모든 원 공개 경로→임시 상대 경로, 기대 SHA/크기·실제 복사 상태, source39·입력 결속. 임시 절대경로는 기록하지 않는다.
- `FLAT_PUBLIC_MANIFEST.sha256.txt`: 삭제된 임시 flat bundle의 manifest 원바이트를 보존한 증거다. 실제 원 통계의 위치는 mapping에 따르며 이 파일을 campaign root의 공개 manifest로 해석하지 않는다.
- `README_KO.md`: 해당 실행의 범위와 한계.

이 검사는 공개 충분통계 재집계다. private NPZ·원 target·모델·GPU·checkpoint 재추론·훈련·비용 타이머를 검증한 것으로 확대하지 않는다.

## 합성 준비 검증

`test_provided_audit_entry.py`만 명시 unittest discovery로 실행한다. 원바이트 복사·flat hash coverage, 중복 우회 금지, gzip SHA 변조, 원본 Cauchy 오류 탐지, numeric PASS 후 사후 실패 보존, 기본 모드 gzip 미접근을 작은 임시 합성 fixture로 검사한다.

제공 원본의3 selftest(`valid_macro_average`, `tampered_score`, `duplicate_band`)도 별도 **합성** 증거다. 이3개나 caller 단위검사 개수를 실제451 조건 또는 campaign 전체 CPU suite의 실험 수로 더하지 않는다. 저장소 기본 pytest는 실행하지 않는다.
