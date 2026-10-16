# 충분통계의 무손실 공개 projection

목적은 같은 예측으로부터 두 번 작성된 공개 충분통계 중 내부 사본만 공개 payload에서 제외해 용량을 줄이는 것이다. 원본 파일을 삭제하거나 수정하지 않는다. 모델·점수·선택·실험 상태·registry·기존 과학 소스의 변경이 아니다.

현재 combination writer는 `combinations/{family}/{label}_V_select_error_stats.csv.gz`를 만들고, 표준 wrapper는 같은 예측 archive에서 `predictions/{registered_id}/V_select/PUBLIC_ERROR_STATS.csv.gz`를 별도로 만든다. 두 파일의 run ID와 gzip 생성 헤더가 달라 압축 SHA는 대개 다르다. 이 도구는 단순한 파일명 추정이나 수치 허용오차로 중복을 판단하지 않는다.

## 제외할 수 있는 조건

`publication_projection.py`는 현재 존재하는 내부 `combinations/*/*_error_stats.csv.gz`마다 다음 연결을 확인한다.

1. `PRE_TARGET_CONTRACT.json`의 family ID·output label·registered run ID·split·kind·variant가 내부 receipt 및 표준 manifest와 일치한다. alpha의 alias는 이 봉인 계약에서 얻으며 suffix 추정으로 만들지 않는다.
2. receipt가 `COMPLETE_VALID`이고, 봉인된 `expected_receipt_contract` 전체와 맞는다. fit template의 model SHA를 selected parameter SHA와 혼동하지 않는다.
3. PRE_TARGET의 실제 SHA, REFERENCE의 8개 identity, 표준 manifest identity, 네 개 참조 배열 지문과 다섯 개 archive 배열 지문이 서로 일치한다.
4. receipt와 표준 manifest가 같은 private archive 경로·파일명·SHA·배열 선언 지문을 가리킨다. 실제 V archive의 byte SHA도 계산한다. NPZ를 역직렬화하거나 배열·미래 target 값을 해석하지 않는다.
5. 실제 부모 manifest 파일의 SHA·run/model/archive identity와 selected head artifact SHA를 확인한다. 현재 source 파일이 예전 학습 source와 같다는 주장은 하지 않으며, 학습 당시 receipt와 계약 SHA를 보존한다.
6. 내부/표준 CSV 헤더가 정확히 같은 10열이고, 매 행의 run ID가 각 계약에 맞는다. run ID를 제외한 `split, cohort, origin, channel, horizon_start, horizon_end, sse, sae, count` 9열을 순서 그대로 문자열 byte 단위로 비교한다. `0.25`와 `0.250`도 다른 것으로 처리한다. count가 0인 행도 비교에서 제외하지 않는다.
7. 행 수가 봉인된 origin 수 × channel 수 × horizon band 수와 같다. 검증 도중 공개 근거 파일이 바뀌지 않았는지 SHA를 다시 확인한다.

하나라도 증명하지 못하면 원래 통계 파일을 payload에 유지하고 `UNPROVEN_DUPLICATE_KEPT`를 기록한다. 독립적으로 증명한 다른 파일의 projection은 보존한다. 전체 상태는 실패 항목이 있으면 `FAIL`이며, 실패를 숨기고 게시 준비 완료로 바꾸지 않는다.

이는 이미 존재하는 두 공개 충분통계 사이의 무손실성 검사다. archive byte SHA와 기존 배열 지문 선언의 동일성을 검사하며, 실제 배열 지문 재계산·예측 재채점·독립 과학 감사·모델 재학습을 새로 수행하지 않는다. 과학적 정확성은 기존 독립 감사와 연결해 판단해야 한다. 현재 지원하는 내부 통계 split은 V_select이며 다른 split은 제외하지 않고 실패 기록으로 남긴다.

## 보존 범위

표준 `PUBLIC_ERROR_STATS.csv.gz`, 예측 manifest, REFERENCE/PRE_TARGET, source, receipt, scores, 실패 증거는 유지한다. `publication_check.py`의 기존 기본 제외와 승인된 로컬 전용 두 항목은 그대로 적용한다.

- `diagnostics/initialization_anchor_1792059651372799700`
- `execution_batches/cpu_V_checkpoint_replay_1792060139101125900.json`

중간 figure/report snapshot은 자동으로 제외하지 않는다. `--exclude-prefix`에 명시한 `figures/snapshot_*` 또는 `reports/snapshot_*`만 추가 제외할 수 있다. 실제 prediction·receipt·통계를 이 옵션으로 제외하려 하면 거부한다. protocol 또는 최종 문서가 참조하는 snapshot은 유지하고 최종 상대 링크 검사를 통과해야 한다.

## 불변 산출물

매 실행은 새 `publication_projections/snapshot_*/`를 만든다. 이전 산출물은 덮어쓰지 않는다.

- `PROJECTION.json`: 원 압축·해제 SHA, 대체 표준 파일과 SHA, 9열 공통행 SHA, 행 수, run ID 매핑, receipt/manifest/preseal/부모/모델 계보, 실패 이유와 공개 범위.
- `PAYLOAD_FILES.txt`: repository 기준의 명시적인 공개 파일 목록. 검증된 내부 중복만 제거하고 이 projection 스냅샷 자체도 포함한다.
- `README_KO.md`: 검사 범위와 표준 충분통계 링크.

원본 gzip은 로컬에 보존한다. 표준 파일의 run ID를 선언된 매핑으로 바꾸면 같은 충분통계를 얻을 수 있지만, 원 gzip 헤더까지 포함한 압축 byte의 재구축은 주장하지 않는다. 원 압축/해제 SHA는 당시 사본의 식별·추적용으로 보존한다. 재압축·점수 반올림·통계 축약을 수행하지 않는다.

## 실행과 게시 검사

campaign 디렉터리에서 다음과 같이 새 projection을 생성한다. 실제 private cache는 CLI 인자로만 전달하며 공개 산출물에 절대 경로를 기록하지 않는다.

```powershell
python -B publication_projection.py --campaign . --cache-root "<private-cache-root>"
```

최종 allowlist를 만들기 전에 protocol·최종 문서에 참조되지 않는 중간 snapshot만 명시적으로 추가할 수 있다.

```powershell
python -B publication_projection.py --campaign . --cache-root "<private-cache-root>" --exclude-prefix "figures/snapshot_<explicit-id>"
python -B publication_check.py --campaign . --preflight --file-list "publication_projections/snapshot_<new-id>/PAYLOAD_FILES.txt"
```

두 번째 명령의 기존 preflight helper는 `checks/`에 새 검사 영수증을 쓴다. 읽기 전용 검사만 요청된 경우에는 `publication_check.inspect` API를 사용한다. 이 도구는 `prepare`, Git stage/commit/push를 자동 실행하지 않는다. 최종 실행 출력이 계속 추가되는 동안 만든 projection은 그 시점의 후보 목록이며, 마지막 보고서·그림·출력 동결 후 새 snapshot으로 다시 생성해야 한다.

원 파일당 20 MiB·전체 250 MiB 제한은 최종 `publication_check.py`에서 별도로 검사한다. 이번 projection의 PASS는 최종 게시 검사 PASS나 과학 캠페인 완료를 의미하지 않는다.
