# V8 실행·재현 기록

실제 실행 순서와 예약·PID·시간·소스 hash는 [ledger.json](ledger.json), 실행 명령은 각 `runs/*/receipt.json`과 평가 JSON의 `environment.command`에 있다. 기존 v1–v7 경로·checkpoint·prediction hash는 [reuse_manifest.json](reuse_manifest.json), 새 초기/선택/restart checkpoint와 예측 위치는 `initial_manifest_*.json`, `selected_*.json`, `runs/*/result.json`, `*_eval01.json`에 연결된다. 파일 경로는 이 작업 머신의 실제 경로이며 raw/weights/prediction arrays는 공개하지 않는다. 공개 결과만으로 원자료에 대한 독립 재현이 완료됐다고 주장하지 않는다.

환경: 저장소 루트에서 `.venv/Scripts/python.exe -B`를 사용했다. 설치 버전은 평가의 `environment`에 고정돼 있다. 실행 코드는 `runtime_v8.py`와 `model_v8.py`를 이 폴더에서 import한다. 모델/수치 학습 코드는 TEST 뒤 변경하지 않았다.

승인된 실행 순서는 다음과 같다. **이미 완료된 폴더에서 다시 학습하는 명령이 아니다.** 실행기는 중복 job과 TEST 이후 적합을 차단한다. 새 독립 재현에는 별도 승인·회차·원장이 필요하며 기존 봉인/원장을 삭제해 재실행하지 않는다.

1. 기존 계약·배열·기준 예측 hash 재사용 확인, 이전 연구 보존 hash 기록.
2. CPU 합성 검사 1세션, 총10update. 실제 자료 optimizer smoke 없음.
3. 양쪽 자료의 PCA/두 seed 초기 상태를 준비. 각 자료에서 `run_fits_v8.py --dataset robin` 또는 `--dataset jena`로 사전 고정한 네 설정을 순차 실행. 각 자식 학습은 독립 attempt와 GPU job으로 기록됨.
4. 양쪽의 VAL 선택이 완료된 후 `evaluate_v8.py --seal --dataset ... --selection ... --name ... --reserve-seconds 120`. 이 시점에는 신규 TEST 출력 없음.
5. 고정 평가: `evaluate_v8.py --evaluate --dataset robin --name robin_eval01 --reserve-seconds 600`, 이어서 Jena `--dataset jena --name jena_eval01 --reserve-seconds 1200`. 이후 신규 fit 없음.
6. 아래 CPU 분석·그림·검산. 모델 추론을 반복하지 않고 저장 예측과 수치를 검산한다.

```powershell
.venv/Scripts/python.exe -B research/tsfm_peft_subspace_correction_v8_20260929/analyze_v8.py --name analysis01 --robin-evaluation research/tsfm_peft_subspace_correction_v8_20260929/robin_eval01.json --jena-evaluation research/tsfm_peft_subspace_correction_v8_20260929/jena_eval01.json --reserve-seconds 180
.venv/Scripts/python.exe -B research/tsfm_peft_subspace_correction_v8_20260929/plot_v8.py --job figures01
.venv/Scripts/python.exe -B research/tsfm_peft_subspace_correction_v8_20260929/verify_v8.py --job final_checks01 --output final_checks.json --analysis analysis01/analysis_summary.json --reserve-seconds 300
```

`*_partial.json`은 완료 평가의 중간 저장이며 추가 독립 실행이 아니다. `protocol_initial_v7_cost_copy.json`과 `runtime_before_test_guard.txt`는 [contract_correction01.json](contract_correction01.json)에 기록한 TEST 전 계약 정정의 원본이다. 실제 source receipt상 초기 Robin3fit이 이 원본을 사용했으며, 최초의 첫fit-only 서술은 [contract_correction02.json](contract_correction02.json)으로 정정했다. 과거 해시를 현재 해시로 덮어쓰지 않았다. 검산 실패01/02의 원본·원인·수리도 보존하며 `final_checks.json`은 수리 후 실제 실행 상태를 나타낸다. 위 최초 검산 명령의 실패 역시 CPU 원장에 계상돼 있다.

새 비용은 [cost_decision.json](cost_decision.json)의 명시적 분기에 따라 미실행이다. `cost_v8.py`가 존재한다는 사실은 benchmark 완료를 뜻하지 않는다. 코드/검산 PASS, 과학적 주장 지지, 게시 완료는 서로 다른 상태이며 최종 검산·게시 기록의 실제 status를 확인한다.
