# Appliances loss audit v3 repair memo

상태: source-only 준비. original wall cap이 소진된 뒤 작성했으므로 이 memo와 v3 source는 actual CP/NPZ read, model 실행, GPU 실행, pytest 실행을 수행하지 않았다.

## 보존된 실패 상태

- `APPLIANCES_LOSS_AUDIT_v1.json`: `FAILED_PRESERVED`, `checks=126`, `failed_count=0`, `NameError: loss_result`.
- `APPLIANCES_LOSS_AUDIT_v2.json`: `FAILED_PRESERVED`, `checks=128`, `failed_count=0`, q scope 문자열을 dict로 처리한 adapter 오류.
- `APPLIANCES_LOSS_AUDIT_v3.json`: `FAILED_PRESERVED`, `checks=215`, `failed_count=2`, codec old-hash와 APP shape/dtype hash 혼동 및 missing score compare 오류.
- `APPLIANCES_LOSS_AUDIT_v4.json`: `FAILED_PRESERVED`, `checks=9917`, `failed_count=2406`. 6개 final hash는 old/Bi/APP hash 알고리즘 혼동이고, 2400개 curve failure는 MAE/Huber `objective_loss`를 MSE diagnostic `actual_Y_loss`와 같다고 잘못 요구한 adapter 오류다.
- `APPLIANCES_LOSS_AUDIT_v5.json`: `BUDGET_HELD_PRESERVED`, `checks=5449`, `failed_count=0`, 4/6 condition 처리 후 `CAMPAIGN_WALL_BUDGET`.

v1 source는 실패 뒤 수정되어 실패 당시 파일 내용을 그대로 보존했다고 주장할 수 없다. 해당 output에는 당시 `audit_source_sha256`만 남아 있다.

## v3 source semantics

- queue schema는 `smallk_appliances_loss_queue_v1`이며 `conditions[*].producer_result {path, sha256}`에서 6개 single-fit RESULT를 로드한다.
- single-fit RESULT는 `objective=MSE|MAE|HUBER_DELTA1`를 사용한다. invented `loss`, `original_checkpoint`, `initial_q_state_sha256`, `warmstart_inference` field를 요구하지 않는다.
- old codec hash는 producer의 old algorithm(`name + raw bytes`, insertion order)으로 검산한다.
- q state는 APP shape/dtype algorithm으로 검산하고, 새 training/deployment에서는 unused임을 producer q flags로 확인한다.
- `warmstart_all_splits` 단일 NPZ에서 fit/tune/eval prediction, target, observed, origins, original_LEARNED, mean/std를 검산한다.
- checkpoint 5개 × 6 condition의 FIT/TUNE predictions로 normalized/physical actual-Y MSE/MAE/Huberδ1 및 normalized RAW-teacher metrics를 재계산한다.
- selected inference 6×3=18 outputs의 mask/stats/origins와 scores를 재계산한다.
- curve는 600개 coverage와 finite values를 검산한다. MSE objective만 `objective_loss == actual_Y_loss`가 기대되고, MAE/Huber는 curve만으로 per-update forward replay 없이 objective를 독립 재계산했다고 주장하지 않는다.
- fresh default output은 `APPLIANCES_LOSS_AUDIT_v6.json`이다. 기존 v1-v5 output은 덮어쓰지 않는다.

## 승인 뒤 권장 실행

```powershell
.venv/Scripts/python.exe research/smallk_failure_followup_20261014/validation_v1/audit_appliances_losses_v3.py --root-go --runtime continuation --loss-result .cache/smallk_failure_followup_20261014/appliances_losses/APPLIANCES_QUEUE_v1.json
```

`continuation_runtime_v1.ContinuationLimit`는 runtime 실행 시점에만 lazy import된다.
