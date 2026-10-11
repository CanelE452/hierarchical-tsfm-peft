# 실행·복구 기록

2026-10-11. 작업 위치는 `E:/CODING/proj/hierarchical-tsfm-peft`다. 아래 session/종료코드는 root의 실제 도구 응답을 기록한 것이고, PID·과학 count·시간은 progress/result receipt와 대조했다. 설치·학습·추론을 재실행하는 문서가 아니다.

## 실행 환경과 범위

- Windows PowerShell, `.venv/Scripts/python.exe` → `C:/Users/User/anaconda3/envs/mlts/python.exe`, torch 2.10.0+cu128, RTX4070 12GB.
- 새 연구 폴더와 `.cache/level_k_tradeoff_20261011/`를 사용했다. 기존 tracked code와 이전 결과는 보존했다. history는 기존 파일에 append했다.
- Robin/Jena, 두 seed, PCA·point MLP의 K sweep, 원채널 두 대조, 공식 TS2Vec의 작은 설정을 같은 FORECAST_RIDGE_D8 계약에 연결했다.
- 기존 중간 K MLP 네 checkpoint를 검증 후 재사용했다. 이번 새 학습은 성공한 MLP 8 + TS2Vec 4 fits다. frozen F 역전파, 새 TEST 평가, G 학습은 없다.
- 과학 실행이 끝난 뒤 benchmark를 한 번 실행했다. GPU 작업을 겹치지 않았다. 보고서·최종 감사는 JSON/파일/CPU 해시만 읽는다.

## 과학 실행과 파일 처리 실패

```text
Stage                  Session    PID     Exit  Starts  Complete  Ridge
baseline attempt1        81106   74192       1       1         0      0
baseline attempt2        23607   82348       1       6         5    100
baseline continuation    27563   63384       0       3         3    100
TS2Vec small             28659    6708       0       4         4     40
```

[확인] attempt1은 progress JSON의 atomic replace에서 WinError5가 발생했다. Windows 파일 공유/reader 경합은 가능한 설명이며 원인으로 확정하지 않았다. 실패한 Robin K2 seed92601의 마지막 완료 epoch는62, 기록된 update는992다. epoch 중 미기록 update는0..15로 한정했다. 소스·seal·log·partial·cache를 `failed_attempts/attempt1/`와 보존 manifest에 남겼다. PermissionError에만 10/25/50/100/200ms의 제한된 재시도를 추가하고 새 trajectory로 시작했다.

[확인] attempt2는 다른 agent의 __pycache__ 정리와 storage walk가 겹쳐 WinError3가 발생했다. 완료된 Robin K2/K10 네 fit과 Jena K3 seed92601 한 fit, Robin100 ridge를 보존했다. 미완료 Jena K3 seed92602는 epoch22/352update까지 기록됐으며 추가 미기록 update는0..16이다. 실패 trajectory는 checkpoint 선택에 사용하지 않았다. 완료 cache33파일을 archived SHA와 대조한 뒤 재사용하고 남은 Jena3fit/100ridge만 continuation에서 실행했다. 파일 삭제 정리를 중단하고 storage walk가 __pycache__/일시적으로 사라진 tmp만 제외하도록 고쳤다. 일반 파일 오류와 자원 제한은 계속 실패로 처리한다.

[확인] `recovery_source_diff.json`은 두 복구 전후 science helper AST가 같음을 기록한다. 변경은 atomic JSON retry/storage accounting에 한정했다. continuation은 새3starts/3complete/100ridge와 source seal을 시작·종료 양쪽에서 검사한다. 실패 사본은 삭제하거나 성공 출력으로 덮지 않았다.

성공한 continuation과 TS2Vec의 실제 명령:

```powershell
.venv/Scripts/python.exe -X utf8 research/level_k_tradeoff_20261011/continuation.py --device cuda *> research/level_k_tradeoff_20261011/continuation_science.log
.venv/Scripts/python.exe -X utf8 research/level_k_tradeoff_20261011/method_run.py --device cuda *> research/level_k_tradeoff_20261011/method_science.log
```

```text
Stage                   Wall seconds   Process CPU seconds
baseline attempt1              12.71                 48.59
baseline attempt2             358.96               1244.91
baseline continuation         322.62               1119.56
baseline total                694.29               2413.06
TS2Vec small                  404.18                395.08
science total                1098.46               2808.14
```

[확인] 실패 포함 물리14fit starts/성공12fits/240ridge다. 성공한 새 학습은34560updates, 실패 중 소비된 update까지 합치면35904..35935다. 중간 K 재사용4fits는 새 학습수에서 제외한다. 위 시간은 science stage의 합이며 조사·구현·검증·benchmark·보고서 작성 시간을 포함하지 않는다. baseline `stage_peak_rss_bytes_max`는 stage 종료/실패 snapshot의 최대다. 연속 관측 RSS peak가 아니다. TS2Vec science 종료 RSS도 연속 peak로 해석하지 않는다.

## 전체 추론 비용

```powershell
.venv/Scripts/python.exe -X utf8 research/level_k_tradeoff_20261011/benchmark.py --result research/level_k_tradeoff_20261011/result.json --method-result research/level_k_tradeoff_20261011/method_result.json --device cuda *> research/level_k_tradeoff_20261011/benchmark.log
```

[확인] session45854, exit0. 52독립 process cases, B1/B32, 각각 warmup5/timed20, 총260warmup/1040timed다. 표준화·CPU/GPU 전송·E·F median48·D·CPU 출력·CUDA sync를 포함한다. startup/import/model load/디스크/window 추출·초기 결측 탐지/채점/hash는 제외한다. 최적화된 deployment나 전체 GPU/학습 메모리 비용으로 확대하지 않는다. CUDA allocated/reserved와 sampled CPU RSS는 따로 기록했다.

## 검증·보고서

```powershell
.venv/Scripts/python.exe -B -X utf8 research/level_k_tradeoff_20261011/verify.py --json *> research/level_k_tradeoff_20261011/verification.attempt1.json
.venv/Scripts/python.exe -X utf8 research/level_k_tradeoff_20261011/seal.py check
.venv/Scripts/python.exe -B -X utf8 research/level_k_tradeoff_20261011/verify.py --json *> research/level_k_tradeoff_20261011/verification.attempt2.json
.venv/Scripts/python.exe -B -X utf8 research/level_k_tradeoff_20261011/analyze.py *> research/level_k_tradeoff_20261011/analysis.log
.venv/Scripts/python.exe -B -X utf8 research/level_k_tradeoff_20261011/write_report.py
```

- 최초 독립 검증 session94505/exit1: 2515checks/3fails/2warnings. 실패3개는 no-TEST substring 검사의 `assert_no_test_origins` guard 오탐1개와 미작성 parent after receipt2개다. 실패 출력과 이전 verifier를 보존했다.
- verifier를 AST/JSON exact detection으로 좁혔다. guard는 허용하고 실제 `.test_origins`/`include_test=True`/양수 TEST call/dict/subscript를 거부하는 fixture를 확인했다. science 소스·예측·metric을 수정하지 않았다.
- parent preservation check exit0: 이전4662artifacts/GitHEAD/append-only history prefix 보존.
- 두 번째 독립 검증 session55505/exit0: 2518checks/0fails/2warnings. warning2는 Jena TS2Vec의 ridgeλ1e-6 하한 선택이다. warning은 숨기거나 추가 sweep으로 제거하지 않았다. `verification.json`은 성공한 attempt2의 byte-identical 사본이다.
- continuation/count/seal fixture17PASS, native TS2Vec CPU fixture7PASS, method 구조 fixture5PASS. 과학 실행 전 확인한 작은 fixture며 본 결과의 성능 검증을 대신하지 않는다.
- analyze 최종 exit0. 그림4개를 PNG/PDF로 저장했고 root가 시각적으로 확인했다. batch 비용 그림에도 TS2Vec을 포함했다.
- write_report exit0. `/root/professor_method_choice`가 최종 보고서를 read-only로 대조해 PASS를 반환했다. 수치, 원채널 두 기준선, seed 평균과 MAE 한계, CPU/GPU 비용 범위, 경량 TS2Vec 설정, 순수 시간/LEVEL/신규성 주장 범위를 확인했다. 해당 review는 `report_review.json`에 보고서 SHA와 함께 기록했다.
- 마지막 `final_audit.py`는 과학 재실행 없이 source/archive/input/report/figure/preservation/execution receipt를 대조하고 `final_verification.json`에 실제 결과를 쓴다.

결과와 해석은 [REPORT_KO.md](REPORT_KO.md), 원시 값은 result/method_result/benchmark, 재현 조건은 protocol/method_protocol, 선택 이유와 문헌 범위는 METHOD_CHOICE_KO에 있다. 커밋·푸시·외부 메시지를 실행하지 않았다.
