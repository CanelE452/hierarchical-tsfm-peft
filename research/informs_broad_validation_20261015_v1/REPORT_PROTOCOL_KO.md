# INFORMS 실험 보고서 compiler 계약

`report_campaign.generate(public_root, cache_root)`는 CPU에서 기존 파일만 읽어 `reports/snapshot_<UTC>_<ns>`에 새로운 한국어 보고서를 만든다. 입력·기존 보고서·registry를 덮어쓰지 않는다. 학습, 모델 추론, T 선택, 자원 승인, commit/push를 수행하지 않는다.

## 원수치의 근거

초기 registry와 해시가 고정된 등록 확장을 대사하고, registry의 과학적 identity를 결과 파일과 연결한다. 실제 saved prediction NPZ의 archive SHA, 다섯 배열의 dtype/shape/SHA, 예측 전에 분리해서 저장한 target/mask/origin/channel reference를 확인한다. `audit_predictions`의 독립 FP64 구현으로 원 채널 macro MSE/MAE를 다시 계산하고 공개 SSE/SAE/count도 재집계한다. 저장 score와 불일치하거나 파일이 없으면 그 숫자를 점수표에서 제외하고 이유를 남긴다. 모듈은 raw 데이터 loader나 foundation model을 열지 않는다.

숫자 유효성과 registry execution 상태는 서로 다른 축이다. 실제 캐시가 재검산되더라도 registry가 PENDING이면 그 사실을 보존하고 `registry_reconciled=False`로 기록한다. 이 상태를 전체 감사 PASS, 실험 완료, 수렴 또는 연구 성공으로 올리지 않는다. .25 screen과 1/3 epoch, 선택 checkpoint와 최종 노출량을 합쳐 쓰지 않는다. seed·cohort·학습 상한이 다른 결과를 가장 좋은 값 하나로 줄이지 않는다.

T는 기존 `SELECTION_LOCK.json`의 locked 상태, run ID와 artifact/route hash가 맞는 경우에만 private 예측 캐시를 읽는다. 잠금이 없거나 틀리면 T 캐시를 열기 전에 제외한다. 읽는 것은 이미 실행된 허용 T 예측이며 새 데이터의 T를 생성하지 않는다. 모든 사용 가능한 V/T 조건을 점수표에 유지한다. E0의 과거 B 개발 결과는 별도 CSV에 남기며 새 V/T와 섞지 않는다.

SCORES.csv의 panel/length/seed/K는 등록 identity, LoRA rank는 실제 LoRA source spec에서 읽는다. execution_status/validity_status는 registry 원 상태이고 독립 재검산 상태는 numeric_audit에 별도로 둔다. 최종 출력은 deterministic_point_forecast로 표시한다. F가 사용되면 내부 quantile 0.5 선택을 foundation_point에 기록하되 decoder·결합 출력까지 조건부 중앙값이라고 부르지 않는다. 사전학습 corpus와 데이터 중복 여부는 독립 감사하지 않았으며 중복 존재/부재를 주장하지 않는다.

## 실험·검증·게시 상태

`SUMMARY.json`은 registry의 execution/training/validity/scientific/publication 집계를 각각 기록한다. 별도로 완료 fit receipt, 실제 선택 artifact 존재·SHA, 숫자 재검산, 실제 replay receipt 재검산, 실제 cost trial 재집계를 기록한다. `RUN_INDEX.csv`에서 명령은 등록된 재현 명령이며 실제 원 실행 argv가 없으면 이를 지어내지 않는다. fit receipt와 source/config/seed/epoch/update/exposure/선택 artifact를 별도 연결한다.

비용은 actual timed trial CSV를 읽어 median/mean을 다시 계산한다. fresh process와 balanced shared-F, batch와 row chunk, 조건 ID를 모두 보존한다. 비용 조건과 정확도 route 연결은 producer의 명시 `accuracy_bindings`만 허용한다. `cost_route_spec`과 `accuracy_route_spec`의 SHA를 재계산하고, 허용된 비용 전용 키 이외의 모든 항목, 실제 V manifest/model/source/data/scaler, 선택 artifact SHA를 대조한다. `__rows` 같은 이름 suffix로 원 실험 ID를 추정하지 않는다. 명시 binding이 없는 측정은 UNBOUND로 보존하고 정확도 결과에 연결하지 않는다. K/C 비율을 측정 속도 향상으로 바꾸어 쓰지 않는다.

GPU 누계는 종료된 journal의 process-wall charge 합계다. 초기 probe처럼 lease ID가 없는 과거 행도 실제 charge에 포함한다. 살아 있는 프로세스의 아직 닫히지 않은 구간은 별도 추정치로 표시하고 종료 합계에 넣지 않는다. 인증 토큰이 있는 private active lease 전체나 token 값을 산출물에 복사하지 않는다. 이것은 실제 GPU 연산 시간·사용률 측정이 아니다.

## 공개 산출물

- `README_KO.md`, `REPORT_KO.md`, `PROFESSOR_BRIEF.md`, `REPRODUCE.md`
- `SCORES.csv`, `RUN_INDEX.csv`, `PREDICTION_MANIFEST.csv`, `COST.csv`, `COSTS.csv`
- `COVERAGE.csv`, `FIT_RECEIPTS.csv`, `MATCHED_COMPARISONS.csv`, `REPLAYS.csv`
- `E0_LEDGER.csv`, `E0_EVALUATION_SCORES.csv`, `CLAIM_EVIDENCE.csv`
- `EVIDENCE_INVENTORY.csv`, `INPUT_ISSUES.csv`, `RESUME_QUEUE.json`, `RESUME_COMMANDS.ps1`
- `SUMMARY.json`, `INPUT_MANIFEST.json`, `REPORT_MANIFEST.json`

이미지는 output SHA가 모두 일치하는 최신 완결 저장 figure snapshot만 상대 경로로 연결한다. 여기서 완결 저장은 전체 실험 완료가 아니다. 경로는 해당 public/private root 내부로 제한하고 외부 탈출은 거부한다. 원 raw 배열, weights, 개인 상담 원문, 원 첨부 MASTER의 비공개 문단, 인증 토큰을 보고서에 복사하거나 연결하지 않는다.

`RESUME_COMMANDS.ps1`은 기존 `INFORMS_ASSET_ROOT` 환경 변수와 선택적인 `INFORMS_CACHE_ROOT`, `INFORMS_PYTHON`을 사용한다. 기존 작업이 실행 중이면 중복 실행하지 않는다. 지원되지 않은 방법/데이터 계약에는 명령을 지어내지 않고 미구현·접근 대기로 남긴다. 실제 남은 GPU/벽시계 상한은 각 runner가 강제하며 이 문서가 추가 자원이나 성공을 보장하지 않는다.

모든 새 학습/등록 명령의 유효 범위는 `PRE_SELECTION_ONLY`다. `SELECTION_LOCK.locked=true`이면 registry 확장, campaign fit, deepen, postfit combine/controls/ordered가 GPU 사용이나 등록 파일 변경 전에 중단한다. 보고서 queue는 같은 campaign에서 재학습하는 명령을 비우고 `NEW_CAMPAIGN_OR_POST_T_DEVELOPMENT_REQUIRED`로 남긴다. 기존 명령은 재현용 pre-selection field에만 보존한다. 이후 연구는 새 campaign/별도 ID와 독립 검증 계약으로 분리하고 기존 T를 학습·hyperparameter 선택에 재사용하지 않는다. 고정된 기존 route의 T 실행·비용·감사는 기존 lock 안에서 수행할 수 있다. E0의 과거 B 개발 연장은 별도 증거이며 고정 E1–E4/T 선택으로 되먹이지 않는다.

## 제한

실행 중 파일이 추가될 수 있으므로 snapshot은 하나의 원자적 시점이 아니라 명시된 읽기 구간의 증거 모음이다. immutable artifact의 exact hash와 snapshot 입력 목록으로 추적한다. 검사 통과는 학습 재실행이나 독립 일반화 증거가 아니다. 내부 past-window bidirectionality/prefix 민감도는 미래 타깃 누출과 구분한다. 역사적으로 본 자료, FP32 역정규화 계보, M5 자산 부재, PeMS08 경과 시간 축을 상세 보고서에 표시한다.

```powershell
& $py -X utf8 -m unittest test_report_campaign -v
& $py -X utf8 report_campaign.py --public-root . --private-root $env:INFORMS_CACHE_ROOT
```
