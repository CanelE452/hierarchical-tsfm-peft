# D1/D2 동일 조건 실제 실행기 준비

현재 상태는 **PREPARATION_ONLY**다. 구현·합성 CPU integration 검사를 실제 old 자료의 예측 성능으로 해석하지 않는다. 준비 중 원 scientific NPZ/CP/모델·GPU를 읽거나 실행하지 않았다.

`run_condition_probes.py --execute`는 실제 입력 bridge의 COMPLETE RESULT, 실제 old 자산 hash 검사 표시, Bi 최종 audit/report/manifest의 파일 SHA 및 phase barrier를 먼저 검사한다. 그 뒤 bridge split NPZ 각 array의 SHA·shape·dtype와 원 RAW의 forecast/target/mask/origins/mean/std를 독립 비교한다. 보호 TEST audit hook도 설치한다.

원 소스의 `engine.predict`는 추론 batch 8이고 MAIN 학습 mini-batch는 4다. runner는 원 batch 8을 사용하고 정규화 재추론 허용오차 `atol=1e-5, rtol=2e-6`을 실행 전에 고정한다. 실패 후 bound를 늘리지 않는다.

D1은 한 프로세스에서 한 자료만 실행한다. FIT RAW forecast−원 preprocessed last만으로 through-origin 및 centered basis를 구한다. identity와 quarter/half/full-C rotation의 일곱 고정 조건 전부를 실제 `F(XB)`에 넣고 저장한다. centered projection의 평균 보정과 원 C anchor 우회를 유지하므로 엄격한 K-only 정보 압축의 증거로 바꾸지 않는다. VAL/EVAL로 rank를 선택하지 않는다.

D2는 한 프로세스에서 한 자료·원 seed만 실행한다. 원 MAIN CP file SHA, metadata, 사전 선언 selected update/state SHA, 모델 구조를 확인하고 strict restore한다. absolute `E(X)`, `z_last`, `F(E(X))`, source GPU D output을 실제 추출·저장한다. delta만을 absolute latent로 사용하지 않는다. 원 GPU D의 저장 MAIN parity를 먼저 통과한 뒤 CPU helper에 absolute bundle을 전달한다. 원 E/F/GELU hidden을 동결하고 출력 head를 Y/RAW/MIXED 목적 각각 FIT 적합·TUNE 정규화 MSE 선택한다.

F의 두 hash는 계산 규칙이 다르다. 원 순서 기반 hash는 `c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874`, Bi sorted key/dtype/shape hash는 `e0e0ab8b3e15a2f965a6c1f6692c49122921253da24cc6cceb457d8a038afd7f`다. 두 기대값 및 실행 전후 동결 여부를 함께 검사한다.

자원은 `followup_runtime.FollowupLimit(tag, gpu=True)`로 현재 Bi의 같은 GPU/CPU/campaign/storage 상한을 계속 사용한다. 원 engine Limit나 run을 재개하지 않는다. helper의 process birth 기준 300초 cap 때문에 실제 명령 하나당 D1 자료 한 개 또는 D2 자료·seed 한 개만 처리한다. 입력 hash/load, 원 source import/model load/hash, E/F/D extraction, CPU head refit, 전체 runner wall을 구분하고 원 source 및 새 source snapshot·terminal receipt를 새 폴더에 보존한다. extraction 측정은 warm repeat latency benchmark가 아니다.

root가 Bi 최종 보고서를 확인하고 phase를 전환한 뒤 실행한다. 아래 `$bridge`는 실제 완료 bridge RESULT의 절대경로로 바꾼다.

```powershell
$bridge = 'E:/CODING/proj/hierarchical-tsfm-peft/.cache/smallk_failure_followup_20261014/input_bridge/<fresh>/RESULT.json'
.venv/Scripts/python.exe research/smallk_failure_followup_20261014/run_condition_probes.py --execute --stage D1 --dataset robin --input-bridge $bridge --tag d1_robin_v1
.venv/Scripts/python.exe research/smallk_failure_followup_20261014/run_condition_probes.py --execute --stage D1 --dataset beijing_aotizhongxin --input-bridge $bridge --tag d1_beijing_v1
.venv/Scripts/python.exe research/smallk_failure_followup_20261014/run_condition_probes.py --execute --stage D2 --dataset robin --seed 92601 --input-bridge $bridge --tag d2_robin_s92601_v1
.venv/Scripts/python.exe research/smallk_failure_followup_20261014/run_condition_probes.py --execute --stage D2 --dataset robin --seed 92602 --input-bridge $bridge --tag d2_robin_s92602_v1
.venv/Scripts/python.exe research/smallk_failure_followup_20261014/run_condition_probes.py --execute --stage D2 --dataset beijing_aotizhongxin --seed 92601 --input-bridge $bridge --tag d2_beijing_s92601_v1
.venv/Scripts/python.exe research/smallk_failure_followup_20261014/run_condition_probes.py --execute --stage D2 --dataset beijing_aotizhongxin --seed 92602 --input-bridge $bridge --tag d2_beijing_s92602_v1
```

완료된 출력은 `.cache/smallk_failure_followup_20261014/condition_probes/<tag>/RESULT.json`과 D1 또는 D2 하위 결과다. 실패한 폴더·부분 결과도 보존한다. 원 MAIN native control은 bridge의 감사된 원 prediction reference를 재사용하며 새 native 학습을 추가하지 않는다. Robin VAL·이미 노출된 Beijing EVAL은 개발 증거다.
