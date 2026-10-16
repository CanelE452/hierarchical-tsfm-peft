# 공개 충분통계의 가까움·중간·먼 horizon 요약

**PREPARED — 실제 연구 통계 분석은 아직 실행하지 않았다.** 최종 독립 감사와 `SCORES.csv` 확정 후 ROOT의 실행 신호가 있어야 실제 gzip을 읽는다. 이 폴더는 기존 학습·모델·registry·원 통계를 수정하지 않는다.

`horizon_summary.py` 한 파일이 표준 `V_select`/`T_final` 점수만 처리한다. 새 방법·모델 선택·사례 선택·CI·유의성 검정·추론을 수행하지 않는다. 원 데이터, private NPZ, checkpoint, GPU에는 접근하지 않는다.

## 입력과 바이트 결속

최종 보고서의 `SCORES.csv`와 같은 보고서의 `INPUT_MANIFEST.json`을 명시한다. 후자의 `inputs` 중 `scope=public`인 `file/sha256/bytes`가 공식 prediction manifest와 원 `PUBLIC_ERROR_STATS.csv.gz`를 결속해야 한다. 원 manifest 자체에 통계 SHA가 없는 경우도 있으므로 별도 공개 입력 영수증을 필수로 받는다.

점수표의 `manifest_relative_path`를 따라가 run/split/cohort/source/config/data/model/scaler와 점수·shape를 대조한다. T는 실제 `SELECTION_LOCK.json`의 잠긴 허용 ID만 처리한다. 같은 run의 저장 예측과 실제 route 재추론이 별도 manifest로 점수표에 있으면 각각 유지하며 좋은 쪽을 선택하거나 합치지 않는다.

통계 pack을 사용하는 clone은 먼저 해당 pack과 alias extension의 restore 명령으로 원 gzip 경로를 복원한다. 이 도구는 archive를 자동 풀거나 없어진 자산을 추정하지 않는다.

## 집계 규칙

구간은 0-based half-open이다.

```text
H    near       middle      far
12   [0,3)      [3,6)       [6,12)
28   [0,7)      [7,14)      [14,28)
48   [0,12)     [12,24)     [24,48)
```

공식 CSV의 run/split/cohort, 정수 origin/start/end/count, 원 nonempty UTF-8 channel 식별자, finite nonnegative SSE/SAE, count 범위와 중복 cell을 검사한다. channel은 숫자 index로 다시 해석하지 않는다. 원 origin×channel×band의 관측0 cell도 필요하며 count0이면 SSE/SAE0이어야 한다.

각 band에서 `math.fsum`으로 채널별 SSE·SAE와 count를 모은 다음, 관측 채널별 SSE/count·SAE/count를 균등 평균한다. 전체는 모든 band의 채널별 충분통계를 다시 합쳐 최종 점수의 MSE/MAE/count/origin/채널 수와 대사한다. **band 평균을 전체 오차로 쓰지 않는다.** 정수 coverage는 정확히 비교하고 부동소수 집계 기준은 원 감사와 같은 `atol=1e-9, rtol=1e-7`이다.

관측이 없는 band의 MSE/MAE는 빈 값, 관측·유효 채널 수는0으로 보존한다. horizon은 행 수 단위이며 서로 다른 자료의 물리 시간이 같다고 해석하지 않는다. 표의 `recipe`는 원 필드가 있으면 보존하고 없으면 명시적 `CONFIG_SHA:<config_sha>` 식별자를 쓴다. 학습률·epoch·family/K/seed와 실제 source/config/data/model/scaler는 별도 열과 입력 영수증에 남긴다.

## CLI

저장소 루트에서 `$Campaign`을 캠페인 상대 경로, `$FinalReport`를 실제 최종 보고서 폴더로 설정한다. 아직 없는 최종 파일을 가정해 실행하지 않는다.

```powershell
$Tool = Join-Path $Campaign 'diagnostics/public_horizon_summary_20261016_v1/horizon_summary.py'
python -X utf8 -B $Tool --public-root $Campaign `
  --scores (Join-Path $FinalReport 'SCORES.csv') `
  --input-manifest (Join-Path $FinalReport 'INPUT_MANIFEST.json')
```

기본값은 **메타데이터 검사만** 수행한다. 원 gzip은 크기만 확인하며 열거나 SHA 계산하지 않는다. 모델·통계 분석·출력 snapshot 생성은0이다.

명시 실행 신호 후 같은 명령에 `--execute-report`를 추가한다. 이때만 원 gzip SHA를 확인하고 내용 집계를 실행한다. 새 `snapshot_<ns>` 폴더에 다음 파일을 저장한다.

- `HORIZON_SCORES.csv`: 전체 조건의 세 구간, 정확한 float 표현과 coverage. 원 통계의 행·숫자를 줄이거나 바꾸지 않는다.
- `INPUT_SHA.json`: 최종 점수표·입력 영수증·공식 manifest·원 gzip 결속과 코드 SHA.
- `RESULT.json`: 전체 재집계 일치·실패 경로·실제 완료 범위. 일부 실패 시 다른 경로는 계속 계산하고 전체 상태는 `INCOMPLETE_FAILED_INPUTS`, CLI exit1이다.
- `README_KO.md`: 해당 snapshot의 집계 범위와 한계.

기존 snapshot은 덮어쓰지 않는다. stdout 기본 명령의 성공을 실제 분석 완료로 세지 않는다. 채널별 원 충분통계가 이미 있어 별도 대형 채널 상세 CSV를 중복 출력하지 않는다.

## 현재 합성 검증

이 폴더의 `test_horizon_summary.py`만 명시 unittest discovery로 실행한다. 불균형 mask의 channel-macro와 micro 차이, 전체 재집계, H12/H28/H48, 관측0 band, 중복·누락 cell·다른 cohort·비정상 값·SHA 변조·경로 탈출·기본 모드 gzip 미접근을 합성 fixture로 검사한다. 실제 V/T 자료의 검사·원예측 독립 재채점·전체 campaign CPU suite를 수행한 것으로 세지 않는다.
